import argparse
import pickle
import struct
from pathlib import Path

import pandas as pd
from array_record.python import array_record_module  # ty:ignore[unresolved-import]
from tqdm import tqdm


def pack_speech_data(audio_bytes: bytes, metadata: dict) -> bytes:
    """Concatenate [u32 metadata_len][pickled metadata][raw audio bytes]."""
    serialized = pickle.dumps(metadata)
    return struct.pack("I", len(serialized)) + serialized + audio_bytes


def write_array_record(df: pd.DataFrame, save_path: Path) -> None:
    writer = array_record_module.ArrayRecordWriter(str(save_path), "group_size:1")
    for row in tqdm(df.itertuples(), total=len(df), desc=save_path.name):
        with open(row.path, "rb") as f:
            data = f.read()
        writer.write(pack_speech_data(data, {"label": row.label, "frames": row.frames}))
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_tsv_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    for tsv_path in args.processed_tsv_paths:
        if not tsv_path.exists():
            print(f"Skipping missing TSV: {tsv_path}")
            continue
        df = pd.read_csv(tsv_path, sep="\t")
        save_path = args.save_dir / f"{tsv_path.stem.replace('_processed', '')}.array_record"
        write_array_record(df, save_path)
        print(f"Wrote {save_path}")


if __name__ == "__main__":
    main()
