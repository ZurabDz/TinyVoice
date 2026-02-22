#!/bin/bash

cd "$(dirname "$0")/.."

uv run python scripts/train_bpe_tokenizer.py \
  --tsv_file /home/penguin/data/ka/train_processed.tsv \
  --vocab_size 128 \
  --save_path /home/penguin/data/ka/packed_dataset/