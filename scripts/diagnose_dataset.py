"""Report split integrity and basic audio/text statistics for processed TSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv", type=Path, nargs="+", help="processed TSV files")
    parser.add_argument(
        "--require-disjoint-speakers",
        action="store_true",
        help="fail when client_id is shared across any supplied split",
    )
    args = parser.parse_args()

    frames_by_split: dict[str, set[str]] = {}
    speakers_by_split: dict[str, set[str]] = {}
    for path in args.tsv:
        df = pd.read_csv(path, sep="\t")
        required = {"path", "label", "frames"} - set(df.columns)
        if required:
            raise ValueError(f"{path} is missing columns: {sorted(required)}")
        name = path.stem
        frames_by_split[name] = set(df["path"].astype(str))
        speakers = set(df.get("client_id", pd.Series(dtype=str)).dropna().astype(str))
        speakers_by_split[name] = speakers
        labels = df["label"].fillna("").astype(str)
        print(
            f"{name}: rows={len(df)} speakers={len(speakers)} "
            f"duration_s={df.frames.sum() / 16000:.1f} "
            f"label_chars_mean={labels.str.len().mean():.1f} "
            f"label_chars_max={labels.str.len().max()}"
        )

    names = list(frames_by_split)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            duplicate_files = frames_by_split[left] & frames_by_split[right]
            duplicate_speakers = speakers_by_split[left] & speakers_by_split[right]
            print(
                f"{left} vs {right}: duplicate_files={len(duplicate_files)} "
                f"shared_speakers={len(duplicate_speakers)}"
            )
            if args.require_disjoint_speakers and duplicate_speakers:
                raise SystemExit(
                    f"shared speakers between {left} and {right}: "
                    f"{sorted(duplicate_speakers)[:5]}"
                )


if __name__ == "__main__":
    main()
