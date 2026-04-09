import csv
import pickle
from pathlib import Path

import numpy as np


class Tokenizer:
    """Character tokenizer with explicit <BLANK> (id 0) and <PAD> (last id)."""

    BLANK_TOKEN = "<BLANK>"
    PAD_TOKEN = "<PAD>"

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        chars = self._collect_characters(self.file_path)

        self.char_to_id = {self.BLANK_TOKEN: 0}
        self.id_to_char = {0: self.BLANK_TOKEN}
        for i, ch in enumerate(sorted(chars), start=1):
            self.char_to_id[ch] = i
            self.id_to_char[i] = ch

        pad_id = len(self.char_to_id)
        self.char_to_id[self.PAD_TOKEN] = pad_id
        self.id_to_char[pad_id] = self.PAD_TOKEN

        self.blank_id = 0
        self.pad_id = pad_id
        self.label_pad_token = pad_id
        self.vocab_size = len(self.char_to_id)

    @staticmethod
    def _collect_characters(file_path: Path) -> set[str]:
        chars = set()
        with file_path.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                chars.update(row.get("label") or "")
        return chars

    def encode(self, text: str) -> np.ndarray:
        return np.fromiter(
            (self.char_to_id[ch] for ch in text if ch in self.char_to_id),
            dtype=np.int32,
        )

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        special_ids = {self.blank_id, self.pad_id}
        return "".join(
            self.id_to_char.get(int(i), "")
            for i in ids
            if not skip_special_tokens or int(i) not in special_ids
        )

    def save_tokenizer(self, save_path: Path) -> None:
        save_path = Path(save_path)
        with (save_path / "tokenizer.pkl").open("wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load_tokenizer(path: Path) -> "Tokenizer":
        path = Path(path)
        with path.open("rb") as fh:
            return pickle.load(fh)
