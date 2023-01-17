from datamodule import CommonVoiceDataModule
from model import MiniatureVoice
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
import composer.functional as cf
from pytorch_lightning.callbacks import ModelPruning


model = MiniatureVoice()

model = cf.apply_squeeze_excite(model)


data_module = CommonVoiceDataModule('data/cv-corpus-12.0-2022-12-07/ka', batch_size=4)


trainer = pl.Trainer(accelerator='gpu', gradient_clip_val=1, max_epochs=50, precision=16, check_val_every_n_epoch=2,
accumulate_grad_batches=12, strategy='deepspeed'
# strategy=DeepSpeedStrategy(
#         stage=3,
#         offload_optimizer=True,
#         offload_parameters=True,
#     ),
)
trainer.fit(model, datamodule=data_module)
