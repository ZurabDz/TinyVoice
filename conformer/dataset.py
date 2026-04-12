import functools
import io
import pickle
import struct
from pathlib import Path

import grain
import numpy as np
import soundfile as sf
import soxr
from scipy.signal import fftconvolve


def unpack_speech_data(combined: bytes) -> tuple[dict, bytes]:
    """Inverse of scripts/generate_packed_data.py:pack_speech_data."""
    metadata_len = struct.unpack("I", combined[:4])[0]
    metadata = pickle.loads(combined[4 : 4 + metadata_len])
    return metadata, combined[4 + metadata_len :]


class FilterByDuration(grain.transforms.Filter):
    """Drop examples whose raw audio length is outside the configured window."""

    def __init__(self, sample_rate: int, min_sec: float, max_sec: float):
        self.min_frames = int(min_sec * sample_rate)
        self.max_frames = int(max_sec * sample_rate)

    def filter(self, element: bytes) -> bool:
        metadata, _ = unpack_speech_data(element)
        return self.min_frames <= metadata["frames"] <= self.max_frames


class DecodeAndTokenize(grain.transforms.Map):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def map(self, element: bytes):
        metadata, audio_bytes = unpack_speech_data(element)
        with io.BytesIO(audio_bytes) as fh:
            audio, _ = sf.read(fh, dtype="float32")
        return {
            "audio": np.asarray(audio, dtype=np.float32),
            "label": self.tokenizer.encode(metadata["label"]),
        }


class SpeedPerturb(grain.transforms.RandomMap):
    def __init__(self, sample_rate: int, speed_range=(0.85, 1.15)):
        self.sample_rate = sample_rate
        self.speed_min, self.speed_max = speed_range

    def random_map(self, element, rng: np.random.Generator):
        speed = rng.uniform(self.speed_min, self.speed_max)
        if abs(speed - 1.0) > 0.01:
            element["audio"] = soxr.resample(
                element["audio"],
                int(self.sample_rate * speed),
                self.sample_rate,
                quality=soxr.HQ,
            ).astype(np.float32, copy=False)
        return element


class AddNoise(grain.transforms.RandomMap):
    def __init__(self, prob: float = 0.5, min_snr_db: float = 10.0, max_snr_db: float = 40.0):
        self.prob = prob
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

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
    def __init__(self, sample_rate: int, prob: float = 0.3, rt60_range=(0.1, 0.6)):
        self.sample_rate = sample_rate
        self.prob = prob
        self.rt60_min, self.rt60_max = rt60_range

    def random_map(self, element, rng: np.random.Generator):
        if rng.random() >= self.prob:
            return element
        audio = element["audio"]
        rt60 = rng.uniform(self.rt60_min, self.rt60_max)
        n = int(rt60 * self.sample_rate)
        impulse = np.exp(-6.9 * np.arange(n, dtype=np.float32) / max(n, 1))
        impulse /= impulse.sum()
        element["audio"] = fftconvolve(audio, impulse, mode="full")[: len(audio)].astype(
            np.float32
        )
        return element


class FitsFixedShape(grain.transforms.Filter):
    """Drop post-augment samples that no longer fit the fixed bucket."""

    def __init__(self, audio_frames_max: int, label_length_max: int):
        self.audio_frames_max = audio_frames_max
        self.label_length_max = label_length_max

    def filter(self, element) -> bool:
        return (
            0 < len(element["audio"]) <= self.audio_frames_max
            and 0 < len(element["label"]) <= self.label_length_max
        )


def _pad_and_collate(batch, audio_frames_max: int, label_length_max: int, pad_id: int):
    """Pad an entire batch to the fixed `(audio_frames_max, label_length_max)` shape."""
    n = len(batch)
    audios = np.zeros((n, audio_frames_max), dtype=np.float32)
    labels = np.full((n, label_length_max), pad_id, dtype=np.int32)
    audio_lengths = np.empty(n, dtype=np.int32)
    label_lengths = np.empty(n, dtype=np.int32)
    for i, item in enumerate(batch):
        a, lbl = item["audio"], item["label"]
        audios[i, : len(a)] = a
        labels[i, : len(lbl)] = lbl
        audio_lengths[i] = len(a)
        label_lengths[i] = len(lbl)
    return audios, labels, audio_lengths, label_lengths


def _open_source(data_dir: str, split: str) -> grain.sources.ArrayRecordDataSource:
    return grain.sources.ArrayRecordDataSource(
        str(Path(data_dir) / f"{split}.array_record")
    )


def build_data_sources(args, eval_split: str = "dev"):
    duration_filter = FilterByDuration(
        sample_rate=args.sampling_rate,
        min_sec=args.min_audio_seconds,
        max_sec=args.max_audio_seconds,
    )
    train = grain.MapDataset.source(_open_source(args.data_dir, "train")).filter(duration_filter)
    eval_ = grain.MapDataset.source(_open_source(args.data_dir, eval_split)).filter(duration_filter)
    steps_per_epoch = len(train) // args.batch_size
    return train, eval_, steps_per_epoch


def _augmented(dataset, args, seed: int):
    if args.enable_speed_perturb:
        dataset = dataset.random_map(SpeedPerturb(args.sampling_rate), seed=seed)
    if args.enable_additive_noise:
        dataset = dataset.random_map(AddNoise(), seed=seed + 1)
    if args.enable_reverb:
        dataset = dataset.random_map(AddReverb(args.sampling_rate), seed=seed + 2)
    return dataset


def build_train_loader(train_source, tokenizer, args, num_epochs: int):
    dataset = (
        train_source.repeat(num_epochs=num_epochs)
        .shuffle(seed=42)
        .map(DecodeAndTokenize(tokenizer))
    )
    dataset = _augmented(dataset, args, seed=1234)
    iter_ds = (
        dataset.to_iter_dataset()
        .filter(FitsFixedShape(args.audio_frames_max, args.label_length_max))
        .batch(
            batch_size=args.batch_size,
            drop_remainder=True,
            batch_fn=_make_collator(args, tokenizer),
        )
    )
    return iter_ds.mp_prefetch(
        grain.MultiprocessingOptions(
            num_workers=args.mp_prefetch_workers,
            per_worker_buffer_size=args.mp_prefetch_buffer,
        )
    )


def build_eval_loader(eval_source, tokenizer, args):
    iter_ds = (
        eval_source.map(DecodeAndTokenize(tokenizer))
        .to_iter_dataset()
        .filter(FitsFixedShape(args.audio_frames_max, args.label_length_max))
        .batch(
            batch_size=args.batch_size,
            drop_remainder=True,
            batch_fn=_make_collator(args, tokenizer),
        )
    )
    return iter_ds.mp_prefetch(
        grain.MultiprocessingOptions(
            num_workers=max(args.mp_prefetch_workers // 2, 1),
            per_worker_buffer_size=args.mp_prefetch_buffer,
        )
    )


def _make_collator(args, tokenizer):
    return functools.partial(
        _pad_and_collate,
        audio_frames_max=args.audio_frames_max,
        label_length_max=args.label_length_max,
        pad_id=int(tokenizer.label_pad_token),
    )
