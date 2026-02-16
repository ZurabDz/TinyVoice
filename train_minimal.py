import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from conformer.tokenizer import Tokenizer
from pathlib import Path
from conformer.config import DataConfig, TrainingConfig, ConformerConfig
from conformer.model import ConformerEncoder
from flax import nnx
import optax
import grain
from conformer.dataset import batch_fn, ProcessAudioData, unpack_speech_data
import jax
import jax.numpy as jnp
import jax.profiler
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

conformer_config = ConformerConfig()
model = ConformerEncoder(
    token_count=len(tokenizer.id_to_char), 
    num_layers=conformer_config.num_encoder_layers,
    d_model=conformer_config.encoder_dim,
    dtype=jnp.bfloat16
)

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


# Configure prefetching with threads (not multiprocess to avoid GPU init issues)
read_options = grain.ReadOptions(
    num_threads=data_config.worker_count,
    prefetch_buffer_size=data_config.prefetch_buffer_size * data_config.batch_size
)

processed_train_dataset = (
    map_train_audio_dataset
    .shuffle(seed=42)
    .map(ProcessAudioData(tokenizer))
    .batch(batch_size=data_config.batch_size, batch_fn=functools.partial(batch_fn, bucket_sizes=data_config.bucket_sizes))
    .to_iter_dataset(read_options=read_options)
)

processed_test_dataset = (
    map_test_audio_dataset
    .map(ProcessAudioData(tokenizer))
    .batch(batch_size=data_config.batch_size, batch_fn=functools.partial(batch_fn, bucket_sizes=data_config.bucket_sizes))
    .to_iter_dataset(read_options=read_options)
)

@jax.jit(static_argnums=(0, 2), donate_argnums=(1, 3))
def train_step(model_graphdef, model_state, optimizer_graphdef, optimizer_state, padded_audios, padded_labels, frames, label_lengths):
    """Training step using Functional API"""
    model = nnx.merge(model_graphdef, model_state)
    optimizer = nnx.merge(optimizer_graphdef, optimizer_state)
    
    def loss_fn(model):
        logits, real_times = model(padded_audios, training=True, inputs_lengths=frames)
        
        logit_paddings = (jnp.arange(logits.shape[1]) >= real_times[:, None]).astype(jnp.float32)
        label_paddings = (jnp.arange(padded_labels.shape[1]) >= label_lengths[:, None]).astype(jnp.float32)
        
        loss = optax.ctc_loss(logits.astype(jnp.float32), logit_paddings, padded_labels, label_paddings, blank_id=tokenizer.blank_id).mean()
        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model=model, grads=grads)
    
    _, new_model_state = nnx.split(model)
    _, new_optimizer_state = nnx.split(optimizer)
    
    return loss, new_model_state, new_optimizer_state

# Initial Split
model_graphdef, model_state = nnx.split(model)
optimizer_graphdef, optimizer_state = nnx.split(optimizer)

# Pre-compilation (Warmup)
print("Pre-compiling for all buckets...")
for b_frames, b_label in tqdm(data_config.bucket_sizes, desc="Compiling buckets"):
    # Dummy inputs for compilation
    d_audios = jnp.zeros((data_config.batch_size, b_frames), dtype=jnp.float32)
    d_labels = jnp.zeros((data_config.batch_size, b_label), dtype=jnp.int32)
    d_frames = jnp.full((data_config.batch_size,), b_frames, dtype=jnp.int32)
    d_label_lengths = jnp.full((data_config.batch_size,), b_label, dtype=jnp.int32)
    
    # Trigger JIT
    _, model_state, optimizer_state = train_step(
        model_graphdef, model_state, 
        optimizer_graphdef, optimizer_state, 
        d_audios, d_labels, d_frames, d_label_lengths
    )
    print(f"compiling {b_frames} {b_label} is done")

# Training Loop
global_step = 0

for epoch in range(train_config.num_epochs):
    train_loss_sum = 0.0
    train_steps = 0
    
    pbar = tqdm(processed_train_dataset, desc=f"Epoch {epoch+1}/{train_config.num_epochs}")
    for element in pbar:
        # 1. Start profiling at global_step 10
        # if global_step == 10:
        #     print("Starting JAX trace...")
        #     jax.profiler.start_trace("./logs")
            
        padded_audios, frames, padded_labels, label_lengths = element
        loss, model_state, optimizer_state = train_step(
            model_graphdef, model_state, 
            optimizer_graphdef, optimizer_state, 
            padded_audios, padded_labels, frames, label_lengths
        )
        
        # 2. Stop profiling at global_step 20
        # if global_step == 20:
        #     print("Stopping JAX trace...")
        #     jax.profiler.stop_trace()
            
        train_loss_sum += loss # Keep as JAX array for now to avoid sync
        train_steps += 1
        global_step += 1
        
        # Save checkpoint only when needed
        if mngr.should_save(global_step):
            # We still need a sync here to save, but it's rare (every 100 steps)
            ckpt = {'model': model_state, 'optimizer': optimizer_state}
            mngr.save(global_step, ckpt)
        
        # Update tqdm only every 10 steps to reduce CPU-GPU sync
        if global_step % 10 == 0:
            avg_train_loss = float(train_loss_sum) / train_steps
            pbar.set_postfix({"train_loss": f"{avg_train_loss:.2f}", "step": global_step})
