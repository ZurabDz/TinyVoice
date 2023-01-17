import torch.nn as nn

class SquaredCTCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(reduction='none')

    def forward(self, input, target, input_lengths, target_lengths):
        per_sample_loss = self.ctc_loss(input, target, input_lengths, target_lengths)
        per_sample_loss = per_sample_loss ** 2
        return per_sample_loss.mean()