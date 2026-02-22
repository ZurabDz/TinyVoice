from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from typing import Union


class Tokenizer:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.char_to_id = {}
        self.id_to_char = {}
        self.blank_id = None
        self._construct_tokenizer()

    def encode(self, text: str):
        ids = []
        for ch in text:
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
        return np.array(ids)

    def decode(self, ids):
        return [self.id_to_char[_id] for _id in ids]

    def _construct_tokenizer(self):
        df = pd.read_csv(self.file_path, sep="\t")
        all_words = []
        for words in df["label"].str.split():
            all_words.extend(words)

        combined_words = sorted(set(" ".join(all_words)))

        self.id_to_char[0] = "<BLANK>"
        self.blank_id = 0
        self.padding_id = 0

        self.char_to_id["<BLANK>"] = 0

        for i, char in enumerate(combined_words, 1):
            self.char_to_id[char] = i
            self.id_to_char[i] = char

    def save_tokenizer(self, save_path: Path):
        pickle.dump(self, open(save_path / "tokenizer.pkl", "wb"))

    @staticmethod
    def load_tokenizer(path: Path):
        return pickle.load(open(path, "rb"))


class HuggingFaceBPETokenizer:
    def __init__(self, tokenizer_path: Path):
        from tokenizers import Tokenizer as HFTokenizer

        self.tokenizer = HFTokenizer.from_file(str(tokenizer_path))
        self.blank_id = self.tokenizer.token_to_id("<BLANK>")
        self.padding_id = self.tokenizer.token_to_id("<PAD>")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> np.ndarray:
        encoding = self.tokenizer.encode(text)
        return np.array(encoding.ids)

    def decode(self, ids: Union[np.ndarray, list]) -> str:
        return self.tokenizer.decode(ids)

    def batch_encode(self, texts: list[str]) -> list[np.ndarray]:
        encodings = self.tokenizer.encode_batch(texts)
        return [np.array(e.ids) for e in encodings]

    @staticmethod
    def load(path: Path) -> "HuggingFaceBPETokenizer":
        return HuggingFaceBPETokenizer(path)

    @staticmethod
    def from_pretrained(save_path: Path) -> "HuggingFaceBPETokenizer":
        tokenizer_file = save_path / "tokenizer.json"
        if tokenizer_file.exists():
            return HuggingFaceBPETokenizer(tokenizer_file)
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_file}")
