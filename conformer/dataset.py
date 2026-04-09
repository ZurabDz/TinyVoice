import functools
import io
import pickle
import struct
from pathlib import Path

import grain
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve


def unpack_speech_data(combined: bytes) -> tuple[dict, bytes]:
    """Inverse of scripts/generate_packed_data.py:pack_speech_data."""
    metadata_len = struct.unpack("I", combined[:4])[0]
    metadata = pickle.loads(combined[4 : 4 + metadata_len])
    return metadata, combined[4 + metadata_len :]


class FilterByDuration(grain.transforms.Filter):
    """Keep examples whose raw audio length falls inside the configured window."""

    def __init__(self, sample_rate=16000, min_sec=1.0, max_sec=12.0):
        self.min_frames = int(min_sec * sample_rate)
        self.max_frames = int(max_sec * sample_rate)

    def filter(self, element: bytes) -> bool:
        metadata, _ = unpack_speech_data(element)
        return self.min_frames <= metadata["frames"] <= self.max_frames


class ProcessAudioData(grain.transforms.Map):
    """Decode packed audio bytes and tokenize the target transcript."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def map(self, element: bytes):
        metadata, audio_bytes = unpack_speech_data(element)
        with io.BytesIO(audio_bytes) as fh:
            audio, _ = sf.read(fh, dtype="float32")

        return {
            **metadata,
            "audio": np.asarray(audio, dtype=np.float32),
            "label": self.tokenizer.encode(metadata["label"]),
        }


class SpeedPerturb(grain.transforms.RandomMap):
    def __init__(self, speed_range=(0.85, 1.15), sample_rate=16000, res_type="soxr_mq"):
        self.speed_min, self.speed_max = speed_range
        self.sample_rate = sample_rate
        self.res_type = res_type

    def random_map(self, element, rng: np.random.Generator):
        speed = rng.uniform(self.speed_min, self.speed_max)
        if abs(speed - 1.0) > 0.01:
            element["audio"] = librosa.resample(
                element["audio"],
                orig_sr=int(self.sample_rate * speed),
                target_sr=self.sample_rate,
                res_type=self.res_type,
            ).astype(np.float32)
        return element


class AddNoise(grain.transforms.RandomMap):
    """Add Gaussian noise at a random SNR."""

    def __init__(self, min_snr_db=10.0, max_snr_db=40.0, prob=0.5):
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.prob = prob

    def random_map(self, element, rng: np.random.Generator):
        if rng.random() >= self.prob:
            return element

        audio = element["audio"]
        signal_power = float(np.mean(audio**2))
        if signal_power <= 0.0:
            return element

        snr_db = rng.uniform(self.min_snr_db, self.max_snr_db)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
        element["audio"] = audio + noise
        return element


class AddReverb(grain.transforms.RandomMap):
    """Convolve with a lightweight synthetic room impulse response."""

    def __init__(self, sample_rate=16000, rt60_range=(0.1, 0.6), prob=0.3):
        self.sample_rate = sample_rate
        self.rt60_min, self.rt60_max = rt60_range
        self.prob = prob

    def random_map(self, element, rng: np.random.Generator):
        if rng.random() >= self.prob:
            return element

        audio = element["audio"]
        rt60 = rng.uniform(self.rt60_min, self.rt60_max)
        impulse_size = int(rt60 * self.sample_rate)
        impulse = np.exp(
            -6.9 * np.arange(impulse_size, dtype=np.float32) / (rt60 * self.sample_rate)
        )
        impulse /= impulse.sum()
        element["audio"] = fftconvolve(audio, impulse, mode="full")[: len(audio)].astype(
            np.float32
        )
        return element


def encoder_output_length(audio_length: int, n_fft: int, hop_length: int) -> int:
    stft_length = (audio_length + (n_fft // 2) * 2 - n_fft) // hop_length + 1
    stft_length = (stft_length - 3) // 2 + 1
    stft_length = (stft_length - 3) // 2 + 1
    return max(int(stft_length), 0)


class FitsBucketsAndCtc(grain.transforms.Filter):
    """Drop examples that no longer fit any bucket or violate CTC length constraints."""

    def __init__(self, bucket_sizes, n_fft: int, hop_length: int):
        self.bucket_sizes = tuple((int(audio), int(label)) for audio, label in bucket_sizes)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

    def filter(self, element) -> bool:
        audio_length = int(len(element["audio"]))
        label_length = int(len(element["label"]))

        if audio_length <= 0 or label_length <= 0:
            return False

        if encoder_output_length(audio_length, self.n_fft, self.hop_length) < label_length:
            return False

        return any(
            audio_length <= bucket_audio_length and label_length <= bucket_label_length
            for bucket_audio_length, bucket_label_length in self.bucket_sizes
        )


def build_data_sources(
    data_dir: str,
    sampling_rate: int,
    batch_size: int,
    eval_split: str = "dev",
):
    """Open the train/eval array records and apply the duration filter."""
    root = Path(data_dir) / "packed_dataset"
    train_source = grain.sources.ArrayRecordDataSource(str(root / "train.array_record"))
    eval_source = grain.sources.ArrayRecordDataSource(str(root / f"{eval_split}.array_record"))

    duration_filter = FilterByDuration(sample_rate=sampling_rate, min_sec=1.0, max_sec=11.0)
    train_dataset = grain.MapDataset.source(train_source).filter(duration_filter)
    eval_dataset = grain.MapDataset.source(eval_source).filter(duration_filter)
    steps_per_epoch = len(train_dataset) // batch_size
    return train_dataset, eval_dataset, steps_per_epoch


def _read_options(args) -> grain.ReadOptions:
    return grain.ReadOptions(
        num_threads=args.worker_count,
        prefetch_buffer_size=args.prefetch_buffer_size * args.batch_size,
    )


def _batch_dataset(dataset, args, tokenizer):
    return dataset.batch(
        batch_size=args.batch_size,
        batch_fn=functools.partial(
            batch_fn,
            bucket_sizes=args.bucket_sizes,
            pad_token_id=tokenizer.label_pad_token,
        ),
    )


def build_train_loader(train_dataset: grain.MapDataset, tokenizer, args, num_epochs: int):
    dataset = train_dataset.repeat(num_epochs=num_epochs).shuffle(seed=42).map(
        ProcessAudioData(tokenizer)
    )
    if args.enable_speed_perturb:
        dataset = dataset.random_map(SpeedPerturb(sample_rate=args.sampling_rate), seed=42)
    if args.enable_additive_noise:
        dataset = dataset.random_map(AddNoise(), seed=42)
    if args.enable_reverb:
        dataset = dataset.random_map(AddReverb(sample_rate=args.sampling_rate), seed=42)
    dataset = dataset.filter(
        FitsBucketsAndCtc(args.bucket_sizes, args.n_fft, args.hop_length)
    ).to_iter_dataset(read_options=_read_options(args))
    return _batch_dataset(dataset, args, tokenizer)


def build_test_loader(test_dataset, tokenizer, args):
    dataset = (
        test_dataset.map(ProcessAudioData(tokenizer))
        .filter(FitsBucketsAndCtc(args.bucket_sizes, args.n_fft, args.hop_length))
        .to_iter_dataset(read_options=_read_options(args))
    )
    return _batch_dataset(dataset, args, tokenizer)


def _select_bucket(data, bucket_sizes):
    max_audio_length = max(len(item["audio"]) for item in data)
    max_label_length = max(len(item["label"]) for item in data)

    for bucket_audio_length, bucket_label_length in bucket_sizes:
        if max_audio_length <= bucket_audio_length and max_label_length <= bucket_label_length:
            return bucket_audio_length, bucket_label_length

    return None


def batch_fn(data, bucket_sizes, pad_token_id: int = 0):
    """Pad a batch into the smallest matching (audio, label) bucket.

    Returns `(audio, labels, audio_lengths, label_lengths)` to keep the training
    and inference paths consistent.
    """

    batch_size = len(data)
    bucket = _select_bucket(data, bucket_sizes)
    if bucket is None:
        max_audio_length = max(len(item["audio"]) for item in data)
        max_label_length = max(len(item["label"]) for item in data)
        raise ValueError(
            "Encountered a batch that does not fit any configured bucket: "
            f"audio_length={max_audio_length}, label_length={max_label_length}, "
            f"bucket_sizes={bucket_sizes}"
        )
    max_audio_length, max_label_length = bucket

    padded_audios = np.zeros((batch_size, max_audio_length), dtype=np.float32)
    padded_labels = np.full((batch_size, max_label_length), pad_token_id, dtype=np.int32)
    audio_lengths = np.zeros(batch_size, dtype=np.int32)
    label_lengths = np.zeros(batch_size, dtype=np.int32)

    for index, item in enumerate(data):
        audio = item["audio"]
        label = item["label"]
        audio_length = len(audio)
        label_length = len(label)

        padded_audios[index, :audio_length] = audio
        padded_labels[index, :label_length] = label
        audio_lengths[index] = audio_length
        label_lengths[index] = label_length

    return padded_audios, padded_labels, audio_lengths, label_lengths
