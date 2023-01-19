from torch.utils.data import Dataset
import torchaudio
import torch.nn as nn
import torch
import pandas as pd
import os.path as osp
import torchaudio.transforms as T
import re
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking
from random import randint

from .utils import from_text


def custom_collate_fn(batch):
    features = [feature for feature, _ in batch]
    labels = [torch.tensor(from_text(label)) for _, label in batch]

    features_lengths = [e.shape[0] for e in features]
    labels_lengths = [e.shape[0] for e in labels]

    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return features, labels, torch.tensor(features_lengths), torch.tensor(labels_lengths)


class CommonVoiceDataset(Dataset):
    def __init__(self, root_dir: str, split: str) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.sample_rate = 16000
        self.hop_length = int(self.sample_rate/(1000/10))
        self.win_length = int(self.sample_rate/(1000/25))
        self.featuriser = torchaudio.transforms.MelSpectrogram(
            self.sample_rate, self.win_length, self.hop_length, n_mels=80)
        self.audio_df = self.__create_dataset()

        self.stretch_factor = 0.8

        self.spec_aug = nn.Sequential(
            TimeStretch(self.stretch_factor, fixed_rate=True),
            FrequencyMasking(freq_mask_param=15),
            # TimeMasking(time_mask_param=5),
        )

    def __create_dataset(self):
        path = osp.join(self.root_dir, f'{self.split}.tsv')
        return pd.read_csv(path, sep='\t', usecols=['path', 'sentence']).to_records('')

    def __featurise_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != 16_000:
            resampler = T.Resample(sample_rate, 16_000, dtype=waveform.dtype)
            waveform = resampler(waveform)

        features = self.featuriser(waveform)
        return features[0].T

    def __len__(self):
        return len(self.audio_df)

    def __getitem__(self, index):
        audio_path, label = self.audio_df[index]
        audio_features = self.__featurise_audio(
            osp.join(self.root_dir, 'clips', audio_path))

        if self.split == 'train':
            if randint(1, 10) < 4:
                audio_features = self.spec_aug(audio_features)

        label = re.sub(r'[^ა-ჰ ]+', '', label)
        return audio_features, label
