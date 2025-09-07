import csv
import os.path as osp

class Tokenizer:
    def __init__(self, vocab: list[str]):
        self.char_to_id = {c: i for i, c in enumerate(vocab)}
        self.id_to_char = {i: c for i, c in enumerate(vocab)}
        self.blank_id = self.char_to_id['<pad>']
        
    @property
    def vocab_size(self):
        return len(self.char_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(c, self.char_to_id['<unk>']) for c in text.lower()]

    def decode(self, ids: list[int]) -> str:
        # CTC decode: remove blanks and duplicates
        last_char_id = self.blank_id
        decoded_chars = []
        for char_id in ids:
            if char_id != self.blank_id and char_id != last_char_id:
                decoded_chars.append(self.id_to_char[char_id])
            last_char_id = char_id
        return "".join(decoded_chars)


def build_tokenizer(data):
    def get_common_voice_vocab(dataset):
        # build vocabulary from training data
        text = " ".join([label for label in dataset])
        vocab = sorted(list(set(text.lower())))
        vocab = ['<pad>', '<unk>'] + vocab
        print(f"Vocabulary size: {len(vocab)}")
        return vocab

    vocab = get_common_voice_vocab(data)
    return Tokenizer(vocab)

