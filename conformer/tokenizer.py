import json
from pathlib import Path


class Tokenizer:
    def __init__(self, vocab: list[str]):
        self.char_to_id = {c: i for i, c in enumerate(vocab)}
        self.id_to_char = {i: c for i, c in enumerate(vocab)}
        self.blank_id = self.char_to_id["<pad>"]

    @property
    def vocab_size(self):
        return len(self.char_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(c, self.char_to_id["<unk>"]) for c in text.lower()]

    def decode(self, ids: list[int]) -> str:
        # CTC decode: remove blanks and duplicates
        last_char_id = self.blank_id
        decoded_chars = []
        for char_id in ids:
            if char_id != self.blank_id and char_id != last_char_id:
                decoded_chars.append(self.id_to_char[char_id])
            last_char_id = char_id
        return "".join(decoded_chars)

    def save(self, path: Path, name: str = "tokenizer.json") -> None:
        with open(path / name, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "char_to_id": self.char_to_id,
                    "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
                    "blank_id": self.blank_id,
                },
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
            vocab = [id_to_char[i] for i in range(len(id_to_char))]
            tokenizer = cls(vocab)
            tokenizer.char_to_id = data["char_to_id"]
            tokenizer.id_to_char = id_to_char
            tokenizer.blank_id = data["blank_id"]
            return tokenizer


def build_tokenizer(data):
    def get_common_voice_vocab(dataset):
        # build vocabulary from training data
        text = " ".join([label for label in dataset])
        vocab = sorted(list(set(text.lower())))
        vocab = ["<pad>", "<unk>"] + vocab
        print(f"Vocabulary size: {len(vocab)}")
        return vocab

    vocab = get_common_voice_vocab(data)
    return Tokenizer(vocab)
