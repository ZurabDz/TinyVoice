from torch.utils.data import Dataset
import torchaudio
import torch.nn as nn
import torch
import pandas as pd

from utils import from_text



def custom_collate_fn(batch):
    features = [feature for feature, _ in batch]
    labels = [torch.tensor(from_text(label)) for _, label in batch]

    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True).unsqueeze(1)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return features, labels 


class RawAudioDataset(Dataset):
    def __init__(self, path: str, *, audio_transforms: bool = True):
        self.path = path
        self.raw_audios_pathes = self.__create_dataset()
        self.audio_transforms = audio_transforms
        self.sample_rate = 16000
        self.hop_length = int(self.sample_rate/(1000/10))
        self.win_length = int(self.sample_rate/(1000/25))
        self.featuriser = torchaudio.transforms.MelSpectrogram(
            self.sample_rate, self.win_length, self.hop_length, n_mels=80)
        self.transforms = nn.Sequential(
            # 80 is the full thing
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            # 256 is the hop size, so 86 is one second
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )

    def __create_dataset(self):
        return pd.read_csv(self.path).to_dict(orient='records')

    def featurise_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)
        assert sample_rate == 16000

        features = self.featuriser(waveform)
        return features[0].T.unsqueeze(0)

    def __len__(self):
        return len(self.raw_audios_pathes)

    def __getitem__(self, index):
        audio_path, label = self.raw_audios_pathes[index].values()
        audio_features = self.featurise_audio(audio_path)

        if self.audio_transforms:
            audio_features = self.transforms(
                audio_features.permute(0, 2, 1)).permute(0, 2, 1).squeeze(0)

        return audio_features, label
