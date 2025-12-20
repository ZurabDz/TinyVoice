from array_record.python import array_record_module
from tqdm import tqdm
import pickle
import struct
import numpy as np
import grain
import librosa
from pathlib import Path
from io import BytesIO
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


class ProcessAudioData(grain.transforms.Map):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def map(self, element: bytes):
        metadata, audio_bytes = unpack_speech_data(element)
        # data = BytesIO(audio_bytes)
        # sig, sr = librosa.load(data, sr=None)
        with io.BytesIO(audio_bytes) as fh:
            sig, sr = sf.read(fh, dtype='float32')
        metadata["audio"] = sig
        metadata["label"] = self.tokenizer.encode(metadata["label"])
        return metadata

def round_up(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple

def batch_fn(data):
    batch_size = len(data)
    max_frames = 235008
    max_label_len = 164

    # Pre-allocate numpy arrays once
    padded_audios = np.zeros((batch_size, max_frames), dtype=np.float32)
    padded_labels = np.zeros((batch_size, max_label_len), dtype=np.int32)
    frames = np.zeros(batch_size, dtype=np.int32)
    label_lengths = np.zeros(batch_size, dtype=np.int32)

    for i, item in enumerate(data):
        audio = item['audio']
        label = item['label']
        
        # Clip if audio is longer than max_frames to prevent crash
        l = min(len(audio), max_frames)
        padded_audios[i, :l] = audio[:l]
        frames[i] = l
        
        ll = min(len(label), max_label_len)
        padded_labels[i, :ll] = label[:ll]
        label_lengths[i] = ll

    return padded_audios, frames, padded_labels, label_lengths