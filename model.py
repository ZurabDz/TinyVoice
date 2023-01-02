import torch.nn as nn
from torchaudio.models.conformer import Conformer
import pytorch_lightning as pl
import torch

class MiniatureVoice(pl.LightningModule):
    def __init__(self):
        super().__init__()
        H = 80
        self.encoder = Conformer(input_dim=H, num_heads=4, ffn_dim=128, num_layers=4, depthwise_conv_kernel_size=31)
        self.decoder = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(H, 34)
        )
        self.criterion = nn.CTCLoss()

    def forward(self, input, lengths):
        encoder_outs, encoder_outs_length = self.encoder(input, lengths)
        output = self.decoder(encoder_outs)
        output = nn.functional.log_softmax(output, dim=-1)
        return output, encoder_outs_length

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        features, labels, features_lengths, labels_lengths = train_batch
        output, output_lengths = self.forward(features, features_lengths)
        
        loss = self.criterion(output.transpose(0, 1), labels, output_lengths, labels_lengths)

        return loss


    def validation_step(self, valid_batch, batch_idx):
        features, labels, features_lengths, labels_lengths = valid_batch
        output, output_lengths = self.encoder(features, features_lengths)

        output = self.decoder(output)
        loss = self.criterion(output.transpose(0, 1), labels, output_lengths, labels_lengths)

        self.log('train_loss', loss)