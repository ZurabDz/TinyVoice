import pytorch_lightning as pl
from dataset import CommonVoiceDataset
from torch.utils.data import DataLoader
from dataset import custom_collate_fn
from typing import Union, Optional
from pathlib import Path


class CommonVoiceDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: Union[str, Path], batch_size: Optional[int] = 4, num_workers: Optional[int] = 6):
        super().__init__()
        if isinstance(root_dir, Path):
            self.root_dir = str(root_dir)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.common_voice_train = CommonVoiceDataset(root_dir, 'train')
        self.common_voice_valid = CommonVoiceDataset(root_dir, 'dev') 
        self.common_voice_test = CommonVoiceDataset(root_dir, 'test')

    def train_dataloader(self):
        return DataLoader(self.common_voice_train, batch_size=self.batch_size,
         num_workers=self.num_workers, collate_fn=custom_collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.common_voice_valid, batch_size=self.batch_size,
         num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.common_voice_test, batch_size=self.batch_size,
         num_workers=self.num_workers, collate_fn=custom_collate_fn)