from array_record.python import array_record_module  # ty:ignore[unresolved-import]
from tqdm import tqdm
import pickle
import struct
import numpy as np
import grain
from pathlib import Path
import io
import soundfile as sf


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
