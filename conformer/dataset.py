import pandas as pd
from pathlib import Path
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
    
    # cpu_device = jax.devices('cpu')[0]
    # mel_filterbank = jax.device_put(mel_filterbank, cpu_device)

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
    
    # waveforms = jax.device_put(waveforms, cpu_device)

    return batched_mel_spectrogram_fn(waveforms)


class AudioDataSource(RandomAccessDataSource):
    def __init__(self, df, tokenizer):
        self.data_df = df[df['duration'] <= 11]

        self.max_duration = self.data_df['duration'].max()
        self.max_tokens = self.data_df['label_token_count'].max()
        self.max_frames = int(self.max_duration * 16_000)

        # self.cpu_device = jax.devices('cpu')[0]
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        sig, sr = librosa.load(self.data_df.iloc[idx]['path'], sr=None)
        tokens = jnp.asarray(self.tokenizer.encode(self.data_df.iloc[idx]['sentence']), dtype=jnp.int32)
        sig = jnp.asarray(sig, dtype=jnp.float32)
        sig_padded = jnp.pad(sig, (0, self.max_frames - sig.shape[0]), mode='constant', constant_values=0)
        tokens_padded = jnp.pad(tokens, (0, self.max_tokens - len(tokens)), mode='constant', constant_values=0)

        return {'audio': sig_padded, 'label': tokens_padded}

    def __len__(self):
        return self.data_df.shape[0]
    

def batch_fn(batch):
    data = {'audios': [], 'labels': [], 'input_lengths': [], 'label_lengths': []}

    for item in batch:
        data['audios'].append(item['audio'])
        data['labels'].append(item['label'])
        data['input_lengths'].append(len(item['audio']))
        data['label_lengths'].append(len(item['label']))

    padded_audios = jnp.asarray(data['audios'], dtype=jnp.float32)
    padded_labels = jnp.asarray(data['labels'], dtype=jnp.float32)

    mel_spectrogram = jax_mel_spectrogram(padded_audios)

    return {
        "inputs": jnp.asarray(mel_spectrogram, dtype=jnp.float16),  # (B, T, F)
        "input_lengths": jnp.asarray(data['input_lengths'], dtype=jnp.int32),
        "labels": jnp.asarray(padded_labels, dtype=jnp.int32),
        "label_lengths": jnp.asarray(data['label_lengths'], dtype=jnp.int32),
    }