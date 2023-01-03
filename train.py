from datamodule import CommonVoiceDataModule
from model import MiniatureVoice
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy


model = MiniatureVoice()
data_module = CommonVoiceDataModule()


trainer = pl.Trainer(accelerator='gpu', gradient_clip_val=1, max_epochs=50, 
# strategy=DeepSpeedStrategy(
        # stage=3,
        # offload_optimizer=True,
        # offload_parameters=True,
    # ),
)
trainer.fit(model, datamodule=data_module)
