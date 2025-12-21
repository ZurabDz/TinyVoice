from tqdm.notebook import tqdm
from conformer.tokenizer import Tokenizer
from conformer.dataset import batch_fn, ProcessAudioData, unpack_speech_data
import grain
from pathlib import Path
from flax import nnx
import numpy as np
import jax
import jax.numpy as jnp
from conformer.model import ConformerModel
from tqdm import tqdm
import optax
import orbax.checkpoint as ocp

tokenizer = Tokenizer.load_tokenizer(Path('/home/penguin/data/tinyvoice/tokenizer/tokenizer.pkl'))


train_audio_source = grain.sources.ArrayRecordDataSource('/home/penguin/data/ka/packed_dataset/train.array_record')
test_audio_source = grain.sources.ArrayRecordDataSource('/home/penguin/data/ka/packed_dataset/test.array_record')


map_train_audio_dataset = grain.MapDataset.source(train_audio_source)
map_test_audio_dataset = grain.MapDataset.source(test_audio_source)


batch_size = 48
steps_per_epoch = len(map_train_audio_dataset) // batch_size
num_epochs = 5

processed_train_dataset = (
    map_train_audio_dataset
    .map(ProcessAudioData(tokenizer))
    .batch(batch_size=batch_size, batch_fn=batch_fn)
    .repeat(num_epochs)
)[:500]

processed_test_dataset = (
    map_test_audio_dataset
    .map(ProcessAudioData(tokenizer))
    .batch(batch_size=batch_size, batch_fn=batch_fn)
)[:1000]


options = ocp.CheckpointManagerOptions(max_to_keep=5, save_interval_steps=100)
manager = ocp.CheckpointManager(
    Path('./checkpoints').absolute(),
    options=options,
    item_names=('model', 'optimizer')
)


model = ConformerModel(token_count=len(tokenizer.id_to_char))

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
        weight_decay=1e-2
    ),
    wrt=nnx.Param
)

if manager.latest_step() is not None:
    latest_step = manager.latest_step()
    print(f"Restoring from checkpoint at step {latest_step}...")
    
    # Define the abstract structure for restoration
    abstract_state = {
        'model': nnx.state(model),
        'optimizer': nnx.state(optimizer),
    }
    
    restored = manager.restore(
        latest_step,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(nnx.state(model)),
            optimizer=ocp.args.StandardRestore(nnx.state(optimizer)),
        )
    )
    nnx.update(model, restored['model'])
    nnx.update(optimizer, restored['optimizer'])


@nnx.jit
def jitted_train(model, optimizer, padded_audios, padded_labels, mask, real_times, label_lengths):
    def loss_fn(model):
        logits = model(padded_audios, mask=mask, training=True)
        
        audio_time_mask = jnp.arange(logits.shape[1]) >= real_times[:, None]
        label_mask = jnp.arange(padded_labels.shape[1]) >= label_lengths[:, None]
        
        loss = optax.ctc_loss(logits, audio_time_mask, padded_labels, label_mask).mean()

        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model=model, grads=grads)

    return loss


@nnx.jit
def jitted_validation(model, padded_audios, padded_labels, mask, real_times, label_lengths):
    logits = model(padded_audios, mask=mask, training=False)
    
    audio_time_mask = jnp.arange(logits.shape[1]) >= real_times[:, None]
    label_mask = jnp.arange(padded_labels.shape[1]) >= label_lengths[:, None]
    
    loss = optax.ctc_loss(logits, audio_time_mask, padded_labels, label_mask).mean()

    return loss


padded_audios, frames, padded_labels, label_lengths = processed_train_dataset[0]


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
    
    # Square mask for attention
    mask = jnp.arange(max_t_final) < real_times[:, None]
    mask = jnp.expand_dims(mask, axis=1).repeat(max_t_final, axis=1)
    
    # MultiHeadAttention mask: (batch, num_heads, q_len, k_len)
    mask = jnp.expand_dims(mask, axis=1).repeat(4, axis=1)

    return mask, real_times


mask, real_times = compute_mask(frames)


z = jitted_train(model, optimizer, padded_audios, padded_labels, mask, real_times, label_lengths)


avg_loss = 0
global_step = 0

if manager.latest_step() is not None:
    global_step = manager.latest_step()

steps_per_epoch = 100

print(f"Steps per epoch: {steps_per_epoch}")

for i, element in enumerate(tqdm(processed_train_dataset)):
    padded_audios, frames, padded_labels, label_lengths = element
    mask, real_times = compute_mask(frames)

    loss = jitted_train(model, optimizer, padded_audios, padded_labels, mask, real_times, label_lengths)

    avg_loss += loss
    global_step += 1
    
    if (i + 1) % 20 == 0:
        print(f"Step {global_step}, Train Loss: {avg_loss / 20:.4f}")
        avg_loss = 0

    # Validation and Checkpointing at the end of each epoch
    if (i + 1) % steps_per_epoch == 0:
        epoch = (i + 1) // steps_per_epoch
        print(f"\nEnd of Epoch {epoch}. Running validation...")
        
        val_loss = 0
        val_steps = 0
        for val_element in tqdm(processed_test_dataset, desc="Validation"):
            v_padded_audios, v_frames, v_padded_labels, v_label_lengths = val_element
            v_mask, v_real_times = compute_mask(v_frames)
            
            v_loss = jitted_validation(model, v_padded_audios, v_padded_labels, v_mask, v_real_times, v_label_lengths)
            val_loss += v_loss
            val_steps += 1
        
        if val_steps > 0:
            avg_val_loss = val_loss / val_steps
            print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")

        # Checkpointing
        print(f"Saving checkpoint at step {global_step}...")
        manager.save(
            global_step,
            args=ocp.args.Composite(
                model=ocp.args.StandardSave(nnx.state(model)),
                optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
            )
        )