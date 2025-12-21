import sys
from pathlib import Path

# Add the parent directory to sys.path to import from conformer
sys.path.append(str(Path(__file__).resolve().parent.parent))

from conformer.dataset import create_array_record_dataset
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_tsv_paths', type=Path, nargs='+', required=True, help="List of processed TSV files")
    parser.add_argument('--save_dir', type=Path, required=True, help="Directory to save packed datasets")

    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    for tsv_path in args.processed_tsv_paths:
        if not tsv_path.exists():
            print(f"Warning: {tsv_path} does not exist. Skipping.")
            continue

        print(f"Packing data from {tsv_path}...")
        df = pd.read_csv(tsv_path, sep='\t')
        
        # Use the name of the TSV file (without extension and "_processed") for the packed filename
        base_name = tsv_path.stem.replace("_processed", "")
        save_path = args.save_dir / f"{base_name}.array_record"
        
        create_array_record_dataset(df, save_path)
        print(f"Packed dataset saved to {save_path}")

if __name__ == "__main__":
    main()