import csv
import os.path as osp
import librosa
import numpy as np
import jax.numpy as jnp
import jax
from grain.sources import RandomAccessDataSource
from jax.scipy.signal import stft
from flax import nnx


def create_mel_filterbank(
    sample_rate=16000,
    n_fft=400,
    n_mels=80,
    fmin=0.0,
    fmax=None,
    dtype=jnp.float32,
):
    fmax = fmax if fmax is not None else sample_rate / 2.0
    mel_fb = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    return jnp.array(mel_fb, dtype=dtype)


@nnx.jit
def jax_mel_spectrogram(
    waveforms,  # Expecting a batch of waveforms
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=80,
    power=2.0,
    log_epsilon=1e-10,
):
    mel_filterbank = create_mel_filterbank(
        sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels
    )
    
    cpu_device = jax.devices('cpu')[0]
    mel_filterbank = jax.device_put(mel_filterbank, cpu_device)

    def process_single_waveform(waveform):
        _, _, stft_matrix = stft(
            waveform,
            fs=sample_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            window='hann',
            boundary='constant',
            padded=False,
        )
        stft_matrix = jnp.transpose(stft_matrix)

        power_spectrogram = jnp.abs(stft_matrix) ** power

        # (Time, Freqs) @ (Freqs, Mels) -> (Time, Mels)
        mel_spectrogram = jnp.dot(power_spectrogram, mel_filterbank.T)

        log_mel_spectrogram = jnp.log(mel_spectrogram + log_epsilon)

        return log_mel_spectrogram

    batched_mel_spectrogram_fn = jax.vmap(process_single_waveform)
    
    waveforms = jax.device_put(waveforms, cpu_device)

    return batched_mel_spectrogram_fn(waveforms)


class AudioDataSource(RandomAccessDataSource):
    def __init__(self, root_path, tokenizer):
        self._reader = csv.DictReader(open(osp.join(root_path, 'train.tsv')), delimiter='\t')
        self._data = [(osp.join(root_path, 'clips_16k', entry['path'].replace('mp3', 'flac')),
                        entry['sentence']) for entry in self._reader]
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        sig, sr = librosa.load(self._data[idx][0], sr=None)
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


    mel_spectrogram = jax_mel_spectrogram(padded_audios)

    return {
        "inputs": jnp.asarray(mel_spectrogram, dtype=jnp.float16),  # (B, T, F)
        "input_lengths": jnp.asarray(input_lengths),
        "labels": jnp.asarray(padded_labels),
        "label_lengths": jnp.asarray(label_lengths),
    }