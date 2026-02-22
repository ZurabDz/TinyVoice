import argparse
from pathlib import Path
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


class TSVCorpus:
    def __init__(self, tsv_path: Path, text_column: str = "label"):
        self.tsv_path = tsv_path
        self.text_column = text_column

    def __iter__(self):
        df = pd.read_csv(self.tsv_path, sep="\t")
        for text in df[self.text_column]:
            yield text


def train_bpe_tokenizer(
    tsv_file: Path,
    save_path: Path,
    vocab_size: int = 5000,
    min_frequency: int = 2,
    text_column: str = "label",
    special_tokens: list[str] | None = None,
):
    if special_tokens is None:
        special_tokens = ["<PAD>", "<UNK>", "<BLANK>"]

    print(f"Reading text from TSV file: {tsv_file}")
    print(f"Using column: {text_column}")

    df = pd.read_csv(tsv_file, sep="\t")
    print(f"Total rows: {len(df)}")

    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=special_tokens,
    )

    corpus = TSVCorpus(tsv_file, text_column)
    tokenizer.train_from_iterator(iterator=corpus, trainer=trainer)

    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path / "tokenizer.json"))
    print(f"Tokenizer saved to {save_path / 'tokenizer.json'}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer using HuggingFace tokenizers"
    )
    parser.add_argument(
        "--tsv_file",
        type=Path,
        required=True,
        help="TSV file with text data",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="label",
        help="Column name containing text (default: label)",
    )
    parser.add_argument("--vocab_size", type=int, default=5000, help="Vocabulary size")
    parser.add_argument(
        "--min_frequency", type=int, default=2, help="Minimum token frequency"
    )
    parser.add_argument(
        "--save_path", type=Path, required=True, help="Path to save tokenizer"
    )
    args = parser.parse_args()

    train_bpe_tokenizer(
        tsv_file=args.tsv_file,
        save_path=args.save_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        text_column=args.text_column,
    )


if __name__ == "__main__":
    main()
