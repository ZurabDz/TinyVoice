from pathlib import Path
import pandas as pd
import numpy as np
import pickle

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
            # If unknown, we skip it.
        return np.array(ids)

    def decode(self, ids: int):
        return [self.id_to_char[_id] for _id in ids]

    def _construct_tokenizer(self):
        df = pd.read_csv(self.file_path, sep='\t')
        all_words = []
        for words in df['label'].str.split():
            all_words.extend(words)

        combined_words = sorted(set(' '.join(all_words)))

        self.id_to_char[0] = '<PADDING>'
        self.padding_id = 0

        self.id_to_char[1] = '<BLANK>'
        self.blank_id = 1

        self.char_to_id['<PADDING>'] = 0
        self.char_to_id['<BLANK>'] = 1
        
        for i, char in enumerate(combined_words, 2):
            self.char_to_id[char] = i
            self.id_to_char[i] = char


    def save_tokenizer(self, save_path: Path):
        pickle.dump(self, open(save_path / 'tokenizer.pkl', 'wb'))

    @staticmethod
    def load_tokenizer(path: Path):
        return pickle.load(open(path, 'rb'))