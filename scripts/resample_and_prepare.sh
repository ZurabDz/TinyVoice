#!/bin/bash

# Default root path
ROOT_PATH="/home/penguin/data/ka"

# Check if a custom root path was provided
if [ ! -z "$1" ]; then
    ROOT_PATH="$1"
fi

echo "Processing Common Voice dataset in $ROOT_PATH"

# Run the processing script for train, dev, and test files
python3 process_common_voice_dataset.py \
    --root_path "$ROOT_PATH" \
    --tsv_filenames train.tsv dev.tsv test.tsv

echo "All files resampled and prepared successfully."
