from datamodule import CommonVoiceDataModule
from model import MiniatureVoice
import pytorch_lightning as pl


model = MiniatureVoice()
data_module = CommonVoiceDataModule()


trainer = pl.Trainer(accelerator='gpu')
trainer.fit(model, datamodule=data_module)
