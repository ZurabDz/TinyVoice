import pytorch_lightning as pl
from dataset import CommonVoiceDataset
from torch.utils.data import DataLoader
from dataset import custom_collate_fn


class CommonVoiceDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.common_voice_train = CommonVoiceDataset('data/cv-corpus-12.0-2022-12-07/ka/', 'train')
        self.common_voice_valid = CommonVoiceDataset('data/cv-corpus-12.0-2022-12-07/ka/', 'dev') 
        self.common_voice_test = CommonVoiceDataset('data/cv-corpus-12.0-2022-12-07/ka/', 'test')

    def prepare_data(self):
        # download
        pass

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(self.common_voice_train, batch_size=4,num_workers=6, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.common_voice_valid, batch_size=4,num_workers=6, collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.common_voice_test, batch_size=4,num_workers=6, collate_fn=custom_collate_fn)