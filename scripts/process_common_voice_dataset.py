import pandas as pd
from pathlib import Path
import librosa
import soundfile as sf
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import string


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=Path, required=True, help="root path for common voice dataset")
parser.add_argument('--output_file_name', type=str, required=False, default='validated_processed.tsv')
parser.add_argument('--max_workers', type=int, required=False, default=min(14, os.cpu_count() or 1))


args = parser.parse_args('--root_path /home/penguin/data/ka/'.split(' '))
# args = parser.parse_args()

MAX_WORKERS = args.max_workers
ROOT_PATH = args.root_path
RESAMPLED_AUDIOS_PATH = args.root_path / 'clips_16k'
RESAMPLED_AUDIOS_PATH.mkdir(exist_ok=True)

dataset = pd.read_csv(ROOT_PATH / 'validated.tsv', sep='\t', usecols=['path', 'sentence'])
dataset['path'] = dataset['path'].apply(lambda path: ROOT_PATH / 'clips' / path)


def resample_and_write(row: tuple, dst_path: Path) -> tuple[str, None | Exception]:
    """
    resamples given audio, writes it to destionation and returns old path, error, new path, label, frames
    """
    try:
        old_path, label = row[1].values
        sig, _ = librosa.load(old_path, sr=16_000)
        saved_path = dst_path / (old_path.stem + '.flac')
        sf.write(str(saved_path), sig, 16_000, format='FLAC')
        return old_path, None, saved_path, label, sig.shape[0]
    except Exception as e:
        return old_path, e, None, None, None
    


new_df_data_source = {'old_path': [], 'error': [], 'path': [], 'label': [], 'frames': []}
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as workers:
    futures = {workers.submit(resample_and_write, row, RESAMPLED_AUDIOS_PATH) 
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
processed_audios = new_df[['path', 'label', 'frames']]
remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
processed_audios['label'] = processed_audios['label'].str.translate(remove_punct_map).str.strip()
processed_audios.to_csv(ROOT_PATH / args.output_file_name, sep='\t', index=False)