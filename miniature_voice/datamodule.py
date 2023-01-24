import pytorch_lightning as pl
from .dataset import CommonVoiceDataset
from torch.utils.data import DataLoader
from typing import Union, Optional
from pathlib import Path
from .collate import MyCollator


class CommonVoiceDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: Union[str, Path], batch_size: Optional[int] = 4, num_workers: Optional[int] = 2):
        super().__init__()
        if isinstance(root_dir, Path):
            self.root_dir = str(root_dir)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.common_voice_train = CommonVoiceDataset(root_dir, 'train', n_mels=80)
        self.common_voice_valid = CommonVoiceDataset(root_dir, 'dev', n_mels=80)
        self.common_voice_test = CommonVoiceDataset(root_dir, 'test', n_mels=80)
        self.custom_collate_fn = MyCollator(root_dir)

    def train_dataloader(self):
        return DataLoader(self.common_voice_train, batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=self.custom_collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.common_voice_valid, batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=self.custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.common_voice_test, batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=self.custom_collate_fn)
