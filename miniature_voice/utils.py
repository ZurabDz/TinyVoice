import itertools
import pandas as pd
import os.path as osp
import re

def remove_punctuation_v2(input_string):
    return re.sub(r'[^\w\s]', '', input_string)

def unique_chars(arr):
    strings = arr.tolist()
    combined = ''.join(strings)
    unique_chars = set(combined)
    return unique_chars

def get_charset(root_dir):
    path = osp.join(root_dir, f'train.tsv')
    data = pd.read_csv(path, sep='\t', usecols=['path', 'sentence'])
    labels = data['sentence'].values
    chareset = ''.join(unique_chars(labels))
    return remove_punctuation_v2(chareset)


def to_text(x, CHARSET):
    x = [k for k, g in itertools.groupby(x)]
    return ''.join([CHARSET[c-1] for c in x if c != 0])


def from_text(x, CHARSET):
    return [CHARSET.index(c)+1 for c in x.lower() if c in CHARSET]
