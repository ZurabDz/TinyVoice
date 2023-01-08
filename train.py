from datamodule import CommonVoiceDataModule
from model import MiniatureVoice
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy


model = MiniatureVoice()
data_module = CommonVoiceDataModule('data/cv-corpus-12.0-2022-12-07/ka')


trainer = pl.Trainer(accelerator='gpu', gradient_clip_val=1, max_epochs=50, precision=16
# strategy=DeepSpeedStrategy(
        # stage=3,
        # offload_optimizer=True,
        # offload_parameters=True,
    # ),
)
trainer.fit(model, datamodule=data_module)
