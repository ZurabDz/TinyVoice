import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_tsv_path", type=Path, required=True)
    parser.add_argument("--tokenizer_save_path", type=Path, required=True)
    return parser.parse_args()


def main():
    from conformer.tokenizer import Tokenizer

    args = parse_args()
    tokenizer = Tokenizer(args.processed_tsv_path)
    args.tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_tokenizer(args.tokenizer_save_path)


if __name__ == "__main__":
    main()
