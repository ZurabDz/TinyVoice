import sys
sys.path.append('..')

from conformer.tokenizer import build_tokenizer, Tokenizer
import pandas as pd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('--processed_tsv_path', type=Path, required=True)
parser.add_argument('--tokenizer_save_path', type=Path, required=True)
# args = parser.parse_args()

args = parser.parse_args('--processed_tsv_path /home/penguin/data/ka/validated_processed.tsv'
' --tokenizer_save_path /home/penguin/data/tokenizer/'.split())

df = pd.read_csv(args.processed_tsv_path, sep='\t')

tokenizer = build_tokenizer(df['label'].values.tolist())

args.tokenizer_save_path.mkdir(exist_ok=True)
tokenizer.save(args.tokenizer_save_path)
