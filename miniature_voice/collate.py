from .utils import get_charset, from_text
import torch


class MyCollator:
    def __init__(self, root_dir):
        self.CHARSET = get_charset(root_dir)

    def __call__(self, batch):
        features = [feature for feature, _ in batch]
        labels = [torch.tensor(from_text(label, self.CHARSET)) for _, label in batch]

        features_lengths = [e.shape[0] for e in features]
        labels_lengths = [e.shape[0] for e in labels]

        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return features, labels, torch.tensor(features_lengths), torch.tensor(labels_lengths)