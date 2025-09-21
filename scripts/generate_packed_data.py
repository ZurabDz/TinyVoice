import sys
sys.path.append('..')

from conformer.dataset import create_array_record_dataset
import pandas as pd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--processed_tsv_path', type=Path, required=True)
parser.add_argument('--packed_dataset_save_path', type=Path, required=True)


# args = parser.parse_args()

# args = parser.parse_args('--processed_tsv_path /home/penguin/data/ka/train_processed.tsv' \
# ' --packed_dataset_save_path /home/penguin/data/packed_dataset/'.split())

args = parser.parse_args('--processed_tsv_path /home/penguin/data/ka/test_processed.tsv' \
' --packed_dataset_save_path /home/penguin/data/packed_dataset/test'.split())

df = pd.read_csv(args.processed_tsv_path, sep='\t')

args.packed_dataset_save_path.mkdir(exist_ok=True)
create_array_record_dataset(df, args.packed_dataset_save_path)