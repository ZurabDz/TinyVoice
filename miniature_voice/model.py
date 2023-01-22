import torch.nn as nn
from torchaudio.models.conformer import Conformer
import pytorch_lightning as pl
import torch
from .utils import to_text
from jiwer import wer
from torch.optim.lr_scheduler import LambdaLR
from .loss import SquaredCTCLoss


class MiniatureVoice(pl.LightningModule):
    def __init__(self, num_classes: int = 34, num_heads: int = 5, ffn_dim: int = 512, num_layers: int = 8, input_dim: int = 128):
        super().__init__()
        self.H = input_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.encoder = Conformer(input_dim=self.H, num_heads=self.num_heads,
                                 ffn_dim=self.ffn_dim, num_layers=self.num_layers, depthwise_conv_kernel_size=31)
        self.decoder = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(self.H, self.num_classes)
        )
        self.criterion = nn.CTCLoss()
        # self.criterion = SquaredCTCLoss()
        self.warmup_steps = 100
        self.current_step = 0

    def forward(self, input, lengths):
        encoder_outs, encoder_outs_length = self.encoder(input, lengths)
        output = self.decoder(encoder_outs)
        output = nn.functional.log_softmax(output, dim=-1, dtype=torch.float32)
        return output, encoder_outs_length

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-5, weight_decay=0.0004)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(
            1, step / self.warmup_steps))
        return [optimizer], [scheduler]
        # return optimizer

    def training_step(self, train_batch, batch_idx):
        features, labels, features_lengths, labels_lengths = train_batch
        output, output_lengths = self.forward(features, features_lengths)

        loss = self.criterion(output.transpose(
            0, 1), labels, output_lengths, labels_lengths)

        self.log('train', loss)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        features, labels, features_lengths, labels_lengths = valid_batch
        output, output_lengths = self.forward(features, features_lengths)

        loss = self.criterion(output.transpose(
            0, 1), labels, output_lengths, labels_lengths)

        # hypothesis = [to_text(e) for e in output.argmax(-1)]
        # ground_truth = [to_text(e) for e in labels]

        self.log('valid_loss', loss)
        # self.log('valid_wer', wer(ground_truth, hypothesis))

    def early_stopping_checkpoint(self):
        # Save a checkpoint at every epoch
        return {
            'epoch': self.current_epoch,
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

    def on_valid_epoch_end(self):
        # Compare current validation loss to the best so far
        if self.current_epoch == 0:
            self.best_loss = float('inf')
        else:
            if self.current_validation_loss < self.best_loss:
                self.best_loss = self.current_validation_loss
            else:
                # If validation loss has increased, stop training
                print(
                    f'Validation loss increased, stopping training at epoch {self.current_epoch}')
                self.trainer.should_stop = True
