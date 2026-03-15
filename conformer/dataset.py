from array_record.python import array_record_module  # ty:ignore[unresolved-import]
from tqdm import tqdm
import pickle
import struct
import numpy as np
import grain
from pathlib import Path
import io
import soundfile as sf
import librosa


def pack_speech_data(audio_bytes, metadata):
    serialized_metadata = pickle.dumps(metadata)
    metadata_len = len(serialized_metadata)
    packed_metadata_len = struct.pack("I", metadata_len)
    combined_data = packed_metadata_len + serialized_metadata + audio_bytes

    return combined_data


def unpack_speech_data(combined_data):
    metadata_len = struct.unpack("I", combined_data[:4])[0]
    metadata_offset = 4 + metadata_len
    serialized_metadata = combined_data[4:metadata_offset]

    parsed_metadata = pickle.loads(serialized_metadata)
    parsed_file_data = combined_data[metadata_offset:]

    return parsed_metadata, parsed_file_data


def create_array_record_dataset(df, save_path: Path):
    writer = array_record_module.ArrayRecordWriter(str(save_path), "group_size:1")

    record_count = 0
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        with open(row.path, "rb") as f:
            data = f.read()

        metadata = {"label": row.label, "frames": row.frames}
        writer.write(pack_speech_data(data, metadata))
        record_count += 1

    writer.close()


class FilterByDuration(grain.transforms.Filter):
    def __init__(self, sample_rate=16000, min_sec=6.0, max_sec=12.0):
        self.min_frames = int(min_sec * sample_rate)
        self.max_frames = int(max_sec * sample_rate)

    def filter(self, element: bytes) -> bool:
        metadata_len = struct.unpack("I", element[:4])[0]
        metadata = pickle.loads(element[4 : 4 + metadata_len])
        frames = metadata["frames"]
        return self.min_frames <= frames <= self.max_frames


class ProcessAudioData(grain.transforms.Map):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def map(self, element: bytes):
        metadata, audio_bytes = unpack_speech_data(element)
        with io.BytesIO(audio_bytes) as fh:
            sig, sr = sf.read(fh, dtype="float32")
        metadata["audio"] = sig
        metadata["label"] = self.tokenizer.encode(metadata["label"])
        return metadata


class SpeedPerturb(grain.transforms.RandomMap):
    def __init__(self, speed_range=(0.85, 1.15), sample_rate=16000):
        self.speed_min = speed_range[0]
        self.speed_max = speed_range[1]
        self.sample_rate = sample_rate

    def random_map(self, element, rng: np.random.Generator):
        speed = rng.uniform(self.speed_min, self.speed_max)
        if abs(speed - 1.0) > 0.01:
            element["audio"] = librosa.resample(
                element["audio"],
                orig_sr=int(self.sample_rate * speed),
                target_sr=self.sample_rate,
            )
        return element


class AddNoise(grain.transforms.RandomMap):
    """Add Gaussian noise at a random SNR between min_snr_db and max_snr_db."""

    def __init__(self, min_snr_db=10.0, max_snr_db=40.0, prob=0.5):
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.prob = prob

    def random_map(self, element, rng: np.random.Generator):
        if rng.random() < self.prob:
            audio = element["audio"]
            signal_power = np.mean(audio ** 2)
            if signal_power > 0:
                snr_db = rng.uniform(self.min_snr_db, self.max_snr_db)
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise = rng.normal(0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
                element["audio"] = audio + noise
        return element


def build_data_sources(data_dir: str, sampling_rate: int, batch_size: int):
    """Create filtered map datasets for train and test splits."""
    train_source = grain.sources.ArrayRecordDataSource(
        data_dir + "/packed_dataset/train.array_record"
    )
    test_source = grain.sources.ArrayRecordDataSource(
        data_dir + "/packed_dataset/test.array_record"
    )
    duration_filter = FilterByDuration(sample_rate=sampling_rate, min_sec=1, max_sec=12.0)
    map_train = grain.MapDataset.source(train_source).filter(duration_filter)
    map_test = grain.MapDataset.source(test_source).filter(duration_filter)
    steps_per_epoch = len(map_train) // batch_size
    return map_train, map_test, steps_per_epoch


def build_train_loader(map_train: grain.MapDataset, tokenizer, args, n_epoch: int):
    """Build the training data loader for a given epoch."""
    import functools

    read_options = grain.ReadOptions(
        num_threads=args.worker_count,
        prefetch_buffer_size=args.prefetch_buffer_size * args.batch_size,
    )
    return (
        map_train.repeat(num_epochs=n_epoch)
        .shuffle(seed=42)
        .map(ProcessAudioData(tokenizer))
        .random_map(SpeedPerturb(sample_rate=args.sampling_rate), seed=42)
        .random_map(AddNoise(), seed=42)
        .to_iter_dataset(read_options=read_options)
        .batch(
            batch_size=args.batch_size,
            batch_fn=functools.partial(
                batch_fn,
                bucket_sizes=args.bucket_sizes,
                pad_token_id=tokenizer.label_pad_token,
            ),
        )
    )


def build_test_loader(map_test, tokenizer, args):
    """Build the test/validation data loader."""
    import functools

    read_options = grain.ReadOptions(
        num_threads=args.worker_count,
        prefetch_buffer_size=args.prefetch_buffer_size * args.batch_size,
    )
    return (
        map_test.map(ProcessAudioData(tokenizer))
        .to_iter_dataset(read_options=read_options)
        .batch(
            batch_size=args.batch_size,
            batch_fn=functools.partial(
                batch_fn,
                bucket_sizes=args.bucket_sizes,
                pad_token_id=tokenizer.label_pad_token,
            ),
        )
    )


def batch_fn(data, bucket_sizes=None, pad_token_id: int = 0):
    batch_size = len(data)

    if bucket_sizes is None:
        # Default fallback if no buckets provided
        max_frames = 235008
        max_label_len = 164
    else:
        # Find the smallest bucket that fits all examples in the batch
        batch_max_frames = max(len(item["audio"]) for item in data)
        batch_max_label = max(len(item["label"]) for item in data)

        # Default to the largest bucket if none fit (though we should probably handle this better)
        max_frames, max_label_len = bucket_sizes[-1]
        for b_frames, b_label in bucket_sizes:
            if batch_max_frames <= b_frames and batch_max_label <= b_label:
                max_frames, max_label_len = b_frames, b_label
                break

    padded_audios = np.zeros((batch_size, max_frames), dtype=np.float32)
    padded_labels = np.full((batch_size, max_label_len), pad_token_id, dtype=np.int32)
    frames = np.zeros(batch_size, dtype=np.int32)
    label_lengths = np.zeros(batch_size, dtype=np.int32)

    for i, item in enumerate(data):
        audio = item["audio"]
        label = item["label"]

        l = min(len(audio), max_frames)
        padded_audios[i, :l] = audio[:l]
        frames[i] = l

        ll = min(len(label), max_label_len)
        padded_labels[i, :ll] = label[:ll]
        label_lengths[i] = ll

    return padded_audios, frames, padded_labels, label_lengths
