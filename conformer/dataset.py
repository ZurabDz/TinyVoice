from array_record.python import array_record_module
from tqdm import tqdm
import pickle
import struct
import numpy as np
import grain
import librosa
from pathlib import Path
from io import BytesIO

# Constants for padding
MAX_AUDIO_LEN = 235008
MAX_LABEL_LEN = 164


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


def create_array_record_dataset(df, root_path: Path):
    example_file_path = root_path / "data.array_record"
    writer = array_record_module.ArrayRecordWriter(
        str(example_file_path), "group_size:1"
    )

    record_count = 0
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        with open(row.path, "rb") as f:
            data = f.read()

        metadata = {"label": row.label, "frames": row.frames}
        writer.write(pack_speech_data(data, metadata))
        record_count += 1

    writer.close()


def batch_fn(batch, tokenizer):
    audios = [item["audio"] for item in batch]
    labels = [item["label"] for item in batch]

    input_lengths = [len(x) for x in audios]
    label_lengths = [len(x) for x in labels]

    # FIXME: fix magic numbers
    padded_audios = np.zeros((len(batch), 235008), dtype=np.float32)
    padded_labels = np.full((len(batch), 164), tokenizer.blank_id, dtype=np.int32)

    for i, (audio, label) in enumerate(zip(audios, labels)):
        padded_audios[i, : len(audio)] = audio
        padded_labels[i, : len(label)] = label

    result = {
        "inputs": padded_audios,
        "input_lengths": np.asarray(input_lengths),
        "labels": np.asarray(padded_labels),
        "label_lengths": np.asarray(label_lengths),
    }

    return result


class ProcessAudioData(grain.transforms.Map):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def map(self, element: bytes):
        metadata, audio_bytes = unpack_speech_data(element)
        data = BytesIO(audio_bytes)
        sig, sr = librosa.load(data, sr=16000)
        metadata["audio"] = sig
        metadata["label"] = self.tokenizer.encode(metadata["label"])
        return metadata
