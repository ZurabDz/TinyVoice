#!/bin/bash

# Default root path
ROOT_PATH="/home/penguin/data/ka"

# Check if a custom root path was provided
if [ ! -z "$1" ]; then
    ROOT_PATH="$1"
fi

echo "Generating ArrayRecords for processed TSVs in $ROOT_PATH"

# Run the packing script for the processed files
python3 generate_packed_data.py \
    --processed_tsv_paths "$ROOT_PATH/train_processed.tsv" "$ROOT_PATH/dev_processed.tsv" "$ROOT_PATH/test_processed.tsv" \
    --save_dir "$ROOT_PATH/packed_dataset"

echo "ArrayRecords generated successfully in $ROOT_PATH/packed_dataset"
