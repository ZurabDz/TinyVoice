import torch
import torch.nn as nn
from torchaudio.models import Conformer
from utils import CHARSET


class GigachadVoice(nn.Module):
    def __init__(self, charset=CHARSET):
        super().__init__()

        self.charset = charset

        C = 64

        self.encode = nn.Sequential(
            nn.Conv2d(1, C, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(C, C, kernel_size=3, stride=2),
            nn.ReLU()
        )

        H = 144

        self.linear = nn.Sequential(
            nn.Linear(C*(((80 - 1) // 2 - 1) // 2), H),
            nn.Dropout(0.1)
        )

        self.conformer = Conformer(
            input_dim=H, num_heads=4, ffn_dim=H * 4, num_layers=16, depthwise_conv_kernel_size=31)

        self.decode = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(H, len(self.charset))
        )

    def forward(self, x, y):
        x = self.encode(x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        y = (y >> 2) - 1
        x = x[:, :torch.max(y)]
        x = self.linear(x)

        x, zz = self.conformer(x, y)
        x = self.decode(x).reshape(x.shape[0], x.shape[1], len(self.charset))

        return torch.nn.functional.log_softmax(x, dim=2).permute(1, 0, 2), zz
