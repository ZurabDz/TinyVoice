import csv
import os.path as osp
import librosa
import numpy as np
import jax.numpy as jnp
from grain.sources import RandomAccessDataSource


class AudioDataSource(RandomAccessDataSource):
    def __init__(self, root_path, tokenizer):
        self._reader = csv.DictReader(open(osp.join(root_path, 'train.tsv')), delimiter='\t')
        self._data = [(osp.join(root_path, 'clips', entry['path']),
                        entry['sentence']) for entry in self._reader]
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        sig, sr = librosa.load(self._data[idx][0], sr=16_000)
        tokens = np.asarray(self.tokenizer.encode(self._data[idx][1]), dtype=np.int32)

        return {'audio': sig, 'label': tokens}

    def __len__(self):
        return len(self._data)
    
    
def batch_fn(batch, audio_config, tokenizer):
    audios = [item['audio'] for item in batch]
    labels = [item['label'] for item in batch]

    input_lengths = [len(x) for x in audios]
    label_lengths = [len(x) for x in labels]

    padded_audios = np.zeros((len(batch), max(input_lengths)), dtype=np.float32)
    padded_labels = np.full((len(batch), max(label_lengths)), tokenizer.blank_id, dtype=np.int32)

    for i, (audio, label) in enumerate(zip(audios, labels)):
        padded_audios[i, :len(audio)] = audio
        padded_labels[i, :len(label)] = label


    mel_spectrogram = librosa.feature.melspectrogram(
        y=padded_audios, sr=audio_config.sampling_rate, n_fft=audio_config.n_fft,
        hop_length=audio_config.hop_length, win_length=audio_config.win_length,
        n_mels=audio_config.n_mels
    )

    log_mel_spectrogram = np.log(mel_spectrogram + 1e-9)


    # Normalize
    mean = np.mean(log_mel_spectrogram)
    std = np.std(log_mel_spectrogram)
    log_mel_spectrogram = (log_mel_spectrogram - mean) / (std + 1e-9)

    return {
        "inputs": jnp.asarray(log_mel_spectrogram.transpose(0, 2, 1)),  # (B, T, F)
        "input_lengths": jnp.asarray(input_lengths),
        "labels": jnp.asarray(padded_labels),
        "label_lengths": jnp.asarray(label_lengths),
    }