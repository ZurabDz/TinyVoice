from tqdm.notebook import tqdm
from conformer.tokenizer import Tokenizer
from conformer.dataset import batch_fn, ProcessAudioData, unpack_speech_data
import grain
from functools import partial
from conformer.conformer_block import ConformerEncoder
from conformer.config import ConformerConfig, TrainingConfig
from flax import nnx
import jax.numpy as jnp
import optax
import jax

conformer_config = ConformerConfig()
train_config = TrainingConfig()


tokenizer = Tokenizer.load('/kaggle/input/tok/pytorch/default/1/tokenizer.json')

train_audio_source = grain.sources.ArrayRecordDataSource('./data/train/data.array_record')
test_audio_source = grain.sources.ArrayRecordDataSource('./data/test/data.array_record')
tokenizer_batch_fn = partial(batch_fn, tokenizer=tokenizer)

map_train_audio_dataset = grain.MapDataset.source(train_audio_source)
map_test_audio_dataset = grain.MapDataset.source(test_audio_source)

processed_train_dataset = (
    map_train_audio_dataset
    .shuffle(seed=42)
    .map(ProcessAudioData(tokenizer))
    .batch(batch_size=48, batch_fn=tokenizer_batch_fn)
    .repeat(5)
)

processed_test_dataset = (
    map_test_audio_dataset
    .map(ProcessAudioData(tokenizer))
    .batch(batch_size=48, batch_fn=tokenizer_batch_fn)
)

model = ConformerEncoder(conformer_config, num_classes=42, rngs=nnx.Rngs(0))

from conformer.train_utils import (
    create_learning_rate_fn,
    train_step,
    eval_step
)

lr_schedule = create_learning_rate_fn(train_config.warmup_steps, conformer_config.encoder_dim)
optimizer = nnx.Optimizer(
    model,
    optax.adamw(
        learning_rate=lr_schedule,
        b1=train_config.beta1,
        b2=train_config.beta2,
        weight_decay=train_config.weight_decay,
    ),
    wrt=nnx.Param
)

loss = train_step(model, optimizer, processed_train_dataset[0])


from clu import metric_writers
from tqdm.notebook import tqdm


logdir = './metrics'

writer = metric_writers.create_default_writer(logdir)
total_loss_accumulator = 0


n_steps_to_save_avg_train_loss = 10
n_steps_for_eval = 1000


for step_count, batch in tqdm(enumerate(processed_train_dataset, 1),
                            total=len(processed_train_dataset), desc="training loop", colour="green"):
    
    loss = train_step(model, optimizer, batch)
    total_loss_accumulator += loss.item()

    if step_count % n_steps_to_save_avg_train_loss == 0:
        avg_loss = total_loss_accumulator / n_steps_to_save_avg_train_loss
        writer.write_scalars(step_count, {'train_loss': avg_loss})
        total_loss_accumulator = 0

    if step_count % n_steps_for_eval == 0:
        total_eval_loss_accumulator = 0
        for eval_batch in tqdm(processed_test_dataset, desc='eval loop', colour='blue', leave=False):
            eval_loss = eval_step(model, batch)
            total_eval_loss_accumulator += eval_loss

        avg_eval_loss = total_eval_loss_accumulator / len(processed_test_dataset)
        writer.write_scalars(step_count, {'eval_loss': avg_eval_loss})
