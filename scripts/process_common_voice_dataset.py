import pandas as pd
from pathlib import Path
import librosa
import soundfile as sf
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import string
import numpy as np

def resample_and_write(row: tuple, dst_path: Path) -> tuple[str, None | Exception]:
    """
    resamples given audio, writes it to destination and returns old path, error, new path, label, frames
    """
    try:
        old_path, label = row[1].values
        sig, _ = librosa.load(old_path, sr=16_000)
        saved_path = dst_path / (old_path.stem + '.flac')
        sf.write(str(saved_path), sig, 16_000, format='FLAC')
        return old_path, None, saved_path, label, sig.shape[0]
    except Exception as e:
        return old_path, e, None, None, None

def process_tsv(tsv_filename: Path, root_path: Path, resampled_path: Path, max_workers: int):
    print(f"Processing {tsv_filename}...")
    # only grab columns we need
    dataset = pd.read_csv(root_path / tsv_filename, sep='\t', usecols=['path', 'sentence'])

    # changing relative path to full path
    dataset['path'] = dataset['path'].apply(lambda path: root_path / 'clips' / path)

    new_df_data_source = {'old_path': [], 'error': [], 'path': [], 'label': [], 'frames': []}
    with ProcessPoolExecutor(max_workers=max_workers) as workers:
        futures = {workers.submit(resample_and_write, row, resampled_path) 
                for row in dataset.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            old_path, error, saved_path, label, frames = future.result()
            if error:
                print(f'something wrong with processing an audio: {old_path}, error is: {error}')

            new_df_data_source['old_path'].append(old_path)
            new_df_data_source['error'].append(error)
            new_df_data_source['path'].append(saved_path)
            new_df_data_source['label'].append(label)
            new_df_data_source['frames'].append(frames)

    new_df = pd.DataFrame.from_dict(new_df_data_source)
    processed_audios = new_df[['path', 'label', 'frames']].copy()
    remove_punct_map = dict.fromkeys(map(ord, string.punctuation + '–—“”„'))
    processed_audios['label'] = processed_audios['label'].str.translate(remove_punct_map).str.strip().str.lower()
    processed_audios['label'] = processed_audios['label'].replace('', np.nan)
    processed_audios.dropna(inplace=True)
    
    output_filename = tsv_filename.stem + "_processed.tsv"
    processed_audios.to_csv(root_path / output_filename, sep='\t', index=False)
    print(f"Finished processing {tsv_filename}. Output saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=Path, required=True, help="root path for common voice dataset")
    parser.add_argument('--tsv_filenames', type=Path, nargs='+', required=True, help="tsv files from which we generate data")
    parser.add_argument('--max_workers', type=int, required=False, default=min(14, os.cpu_count() or 1))

    args = parser.parse_args()

    MAX_WORKERS = args.max_workers
    ROOT_PATH = args.root_path
    RESAMPLED_AUDIOS_PATH = args.root_path / 'clips_16k'
    RESAMPLED_AUDIOS_PATH.mkdir(exist_ok=True)

    for tsv_file in args.tsv_filenames:
        process_tsv(tsv_file, ROOT_PATH, RESAMPLED_AUDIOS_PATH, MAX_WORKERS)