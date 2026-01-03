from tqdm.notebook import tqdm
from conformer.tokenizer import Tokenizer
from conformer.dataset import batch_fn, ProcessAudioData, unpack_speech_data
import grain
from pathlib import Path
from flax import nnx
import numpy as np
import jax
import jax.numpy as jnp
from conformer.model import ConformerEncoder
from tqdm import tqdm
import optax
import orbax.checkpoint as ocp
import os


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1"
NUM_EPOCHS = 1
VAL_EVERY_N_STEPS = 500
PATH = '/home/penguin/data/ka/checkpoints'
TOKENIZER_PATH = '/home/penguin/data/ka/tokenizer/tokenizer.pkl'
TRAIN_ARRAY_PATH = '/home/penguin/data/ka/packed_dataset/train.array_record'
TEST_ARRAY_PATH = '/home/penguin/data/ka/packed_dataset/test.array_record'  

checkpoint_path = Path(PATH)

checkpointer = ocp.CheckpointManager(
    checkpoint_path.absolute(),
    options=ocp.CheckpointManagerOptions(max_to_keep=5)
)

tokenizer = Tokenizer.load_tokenizer(Path(TOKENIZER_PATH))


model = ConformerEncoder(token_count=len(tokenizer.id_to_char))

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-7,
    peak_value=5e-4,
    warmup_steps=1000,
    decay_steps=10000,
    end_value=1e-6
)

optimizer = nnx.Optimizer(
    model,
    optax.adamw(
        learning_rate=lr_schedule,
        b1=0.9,
        b2=0.98,
        weight_decay=1e-2,
        mask=lambda p: jax.tree.map(lambda x: x.ndim > 1, p)
    ),
    wrt=nnx.Param
)

if checkpointer.latest_step() is not None:
    latest_step = checkpointer.latest_step()
    print(f"Restoring from checkpoint at step {latest_step}...")
    
    # Create abstract state for restore template
    abstract_model = nnx.eval_shape(lambda: ConformerEncoder(token_count=len(tokenizer.id_to_char)))
    abstract_optimizer = nnx.eval_shape(lambda: nnx.Optimizer(
        abstract_model,
        optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.98, weight_decay=1e-2),
        wrt=nnx.Param
    ))
    
    restored = checkpointer.restore(
        latest_step,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(nnx.state(abstract_model)),
            optimizer=ocp.args.StandardRestore(nnx.state(abstract_optimizer)),
        )
    )
    nnx.update(model, restored.model)
    nnx.update(optimizer, restored.optimizer)


train_audio_source = grain.sources.ArrayRecordDataSource(TRAIN_ARRAY_PATH)
test_audio_source = grain.sources.ArrayRecordDataSource(TEST_ARRAY_PATH)


map_train_audio_dataset = grain.MapDataset.source(train_audio_source)
map_test_audio_dataset = grain.MapDataset.source(test_audio_source)


processed_train_dataset = (
    map_train_audio_dataset
    .shuffle(seed=42)
    .map(ProcessAudioData(tokenizer))
    .batch(batch_size=24, batch_fn=batch_fn)
    .repeat(1)
)

processed_test_dataset = (
    map_test_audio_dataset
    .map(ProcessAudioData(tokenizer))
    .batch(batch_size=24, batch_fn=batch_fn)
)



def compute_mask(frames):
    # MelSpectrogram: hop_length=160, win_length=400, padded=False
    # T_mel = (T_audio - win_length) // hop_length + 1
    # Conv2dSubSampler: two layers of kernel=3, stride=2, padding='VALID'
    # T_out = (T_in - 3) // 2 + 1
    # T_final = (T_out - 3) // 2 + 1
    
    t_mel = (frames - 400) // 160 + 1
    t_conv1 = (t_mel - 3) // 2 + 1
    t_final = (t_conv1 - 3) // 2 + 1
    
    max_frames = 235008
    max_t_mel = (max_frames - 400) // 160 + 1
    max_t_conv1 = (max_t_mel - 3) // 2 + 1
    max_t_final = (max_t_conv1 - 3) // 2 + 1

    real_times = t_final
    
    # 1D mask: True for valid positions, False for padding
    valid_mask = jnp.arange(max_t_final) < real_times[:, None]  # (batch, max_t_final)
    
    # For MultiHeadAttention: (batch, num_heads, q_len, k_len)
    # True = attend, False = mask out
    # We want to mask keys that are padding, so expand along query dimension
    attention_mask = valid_mask[:, None, None, :]  # (batch, 1, 1, k_len)
    attention_mask = jnp.broadcast_to(attention_mask, (valid_mask.shape[0], 4, max_t_final, max_t_final))

    return attention_mask, real_times


padded_audios, frames, padded_labels, label_lengths = processed_train_dataset[12]
mask, real_times = compute_mask(frames)


@nnx.jit
def jitted_train(model, optimizer, padded_audios, padded_labels, mask, real_times, frames, label_lengths):
    """Training step with gradient computation"""
    def loss_fn(model):
        logits = model(padded_audios, mask=mask, training=True, inputs_lengths=frames)
        
        logit_paddings = (jnp.arange(logits.shape[1]) >= real_times[:, None]).astype(jnp.float32)
        label_paddings = (jnp.arange(padded_labels.shape[1]) >= label_lengths[:, None]).astype(jnp.float32)
        
        loss = optax.ctc_loss(logits, logit_paddings, padded_labels, label_paddings, blank_id=tokenizer.blank_id).mean()
        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model=model, grads=grads)
    return loss

@nnx.jit
def jitted_eval(model, padded_audios, padded_labels, mask, real_times, frames, label_lengths):
    """Evaluation step - no gradient computation"""
    logits = model(padded_audios, mask=mask, training=False, inputs_lengths=frames)
    
    logit_paddings = (jnp.arange(logits.shape[1]) >= real_times[:, None]).astype(jnp.float32)
    label_paddings = (jnp.arange(padded_labels.shape[1]) >= label_lengths[:, None]).astype(jnp.float32)
    
    loss = optax.ctc_loss(logits, logit_paddings, padded_labels, label_paddings, blank_id=tokenizer.blank_id).mean()
    return loss


def run_validation(model, val_dataset):
    """Run validation and return average loss"""
    total_loss = 0.0
    num_batches = 0
    
    for element in val_dataset:
        padded_audios, frames, padded_labels, label_lengths = element
        mask, real_times = compute_mask(frames)
        loss = jitted_eval(model, padded_audios, padded_labels, mask, real_times, frames, label_lengths)
        total_loss += float(loss)
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0



# Warm up model
loss = jitted_train(model, optimizer, padded_audios, padded_labels, mask, real_times, frames, label_lengths)

global_step = checkpointer.latest_step() or 0

print(f"Starting training from step {global_step}")

for epoch in range(NUM_EPOCHS):
    # Training loop
    train_loss_sum = 0.0
    train_steps = 0
    
    pbar = tqdm(processed_train_dataset, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for element in pbar:
        padded_audios, frames, padded_labels, label_lengths = element
        mask, real_times = compute_mask(frames)
        
        loss = jitted_train(model, optimizer, padded_audios, padded_labels, mask, real_times, frames, label_lengths)
        
        train_loss_sum += float(loss)
        train_steps += 1
        global_step += 1
        
        # Update tqdm with running average loss
        avg_train_loss = train_loss_sum / train_steps
        pbar.set_postfix({"train_loss": f"{avg_train_loss:.2f}", "step": global_step})
        
        # # Optional: mid-epoch validation
        # if VAL_EVERY_N_STEPS and global_step % VAL_EVERY_N_STEPS == 0:
        #     val_loss = run_validation(model, processed_test_dataset)
        #     pbar.set_postfix({"train_loss": f"{avg_train_loss:.2f}", "val_loss": f"{val_loss:.2f}", "step": global_step})
    
    # End of epoch validation
    val_loss = run_validation(model, processed_test_dataset)
    avg_train_loss = train_loss_sum / train_steps if train_steps > 0 else 0.0
    
    print(f"\nEpoch {epoch+1} complete - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save checkpoint after each epoch
    checkpointer.save(
        global_step,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(nnx.state(model)),
            optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
        )
    )
    print(f"Checkpoint saved at step {global_step}")