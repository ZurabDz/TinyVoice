import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from conformer.tokenizer import Tokenizer
from pathlib import Path
from conformer.config import DataConfig, TrainingConfig
from conformer.model import ConformerEncoder
from flax import nnx
import optax
import grain
from conformer.dataset import batch_fn, ProcessAudioData, unpack_speech_data
import jax
import jax.numpy as jnp
from pathlib import Path
import functools

# Enable JAX compilation caching
cache_dir = Path.home() / ".cache" / "jax_compilation_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(cache_dir))

from tqdm.auto import tqdm

data_config = DataConfig()
train_config = TrainingConfig() 

tokenizer = Tokenizer.load_tokenizer(Path(data_config.tokenizer_path))

model = ConformerEncoder(token_count=len(tokenizer.id_to_char), dtype=jnp.bfloat16)

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=train_config.lr_init_value,
    peak_value=train_config.lr_peak_value,
    warmup_steps=train_config.lr_warmup_steps,
    decay_steps=train_config.lr_decay_steps,
    end_value=train_config.lr_end_value
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

import orbax.checkpoint as ocp

# Checkpoint Setup
checkpoint_dir = os.path.abspath("./checkpoints")
checkpointer = ocp.PyTreeCheckpointer()
options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=100) # Save every 100 steps for testing
mngr = ocp.CheckpointManager(checkpoint_dir, checkpointer, options)

train_audio_source = grain.sources.ArrayRecordDataSource(data_config.train_data_path)

test_audio_source = grain.sources.ArrayRecordDataSource(data_config.test_data_path)


map_train_audio_dataset = grain.MapDataset.source(train_audio_source)
map_test_audio_dataset = grain.MapDataset.source(test_audio_source)


processed_train_dataset = (
    map_train_audio_dataset
    .shuffle(seed=42)
    .map(ProcessAudioData(tokenizer))
    .batch(batch_size=data_config.batch_size, batch_fn=functools.partial(batch_fn, bucket_sizes=data_config.bucket_sizes))
)

processed_test_dataset = (
    map_test_audio_dataset
    .map(ProcessAudioData(tokenizer))
    .batch(batch_size=data_config.batch_size, batch_fn=functools.partial(batch_fn, bucket_sizes=data_config.bucket_sizes))
)

@nnx.jit
def jitted_train(model, optimizer, padded_audios, padded_labels, frames, label_lengths):
    """Training step with gradient computation"""
    def loss_fn(model):
        logits, real_times = model(padded_audios, training=True, inputs_lengths=frames)
        
        logit_paddings = (jnp.arange(logits.shape[1]) >= real_times[:, None]).astype(jnp.float32)
        label_paddings = (jnp.arange(padded_labels.shape[1]) >= label_lengths[:, None]).astype(jnp.float32)
        
        loss = optax.ctc_loss(logits.astype(jnp.float32), logit_paddings, padded_labels, label_paddings, blank_id=tokenizer.blank_id).mean()
        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model=model, grads=grads)
    return loss

# Pre-compilation (Warmup)
print("Pre-compiling for all buckets...")
for b_frames, b_label in tqdm(data_config.bucket_sizes, desc="Compiling buckets"):
    # Dummy inputs for compilation
    d_audios = jnp.zeros((data_config.batch_size, b_frames), dtype=jnp.float32)
    d_labels = jnp.zeros((data_config.batch_size, b_label), dtype=jnp.int32)
    d_frames = jnp.full((data_config.batch_size,), b_frames, dtype=jnp.int32)
    d_label_lengths = jnp.full((data_config.batch_size,), b_label, dtype=jnp.int32)
    
    # Trigger JIT
    jitted_train(model, optimizer, d_audios, d_labels, d_frames, d_label_lengths)

# Training Loop
epoch = 0
train_loss_sum = 0.0
train_steps = 0
global_step = 0

pbar = tqdm(processed_train_dataset, desc=f"Epoch {epoch+1}/{train_config.num_epochs}")
for element in pbar:
    padded_audios, frames, padded_labels, label_lengths = element
    loss = jitted_train(model, optimizer, padded_audios, padded_labels, frames, label_lengths)
    
    train_loss_sum += float(loss)
    train_steps += 1
    global_step += 1
    
    # Save checkpoint
    ckpt = {'model': nnx.state(model), 'optimizer': nnx.state(optimizer)}
    mngr.save(global_step, ckpt)
    

    
    # Update tqdm with running average loss
    avg_train_loss = train_loss_sum / train_steps
    pbar.set_postfix({"train_loss": f"{avg_train_loss:.2f}", "step": global_step})
