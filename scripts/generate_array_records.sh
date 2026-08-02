#!/bin/bash

# Dataset root contains the processed TSVs; records and tokenizer go in the
# packed subdirectory consumed by TrainingArguments.data_dir.
ROOT_PATH="/home/penguin/data/ka"

# Check if a custom root path was provided
if [ ! -z "$1" ]; then
    ROOT_PATH="$1"
fi

PACKED_PATH="${2:-$ROOT_PATH/packed_dataset}"
mkdir -p "$PACKED_PATH"

echo "Generating ArrayRecords for processed TSVs in $ROOT_PATH"

# Run the packing script for the processed files
python3 generate_packed_data.py \
    --processed_tsv_paths "$ROOT_PATH/train_processed.tsv" "$ROOT_PATH/dev_processed.tsv" "$ROOT_PATH/test_processed.tsv" \
    --save_dir "$PACKED_PATH"

python3 train_tokenizer.py \
    --processed_tsv_path "$ROOT_PATH/train_processed.tsv" \
    --tokenizer_save_path "$PACKED_PATH"

echo "ArrayRecords generated successfully in $ROOT_PATH"
