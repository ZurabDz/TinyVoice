from flax import nnx
import jax.numpy as jnp
from .mel import MelSpectrogram


class Conv2dSubSampler(nnx.Module):
    def __init__(self, d_model, rngs: nnx.Rngs):
        self.module = nnx.Sequential(
            nnx.Conv(in_features=1, out_features=d_model, kernel_size=(3, 3), strides=(2, 2), padding='VALID', rngs=rngs),
            nnx.relu,
            nnx.Conv(in_features=d_model, out_features=d_model, kernel_size=(3, 3), strides=(2, 2), padding='VALID', rngs=rngs),
            nnx.relu
        )

    def __call__(self, x):
        # B, T, D, 1(C)
        output = self.module(x)
        batch_size, subsampled_time, subsampled_freq, d_model = output.shape
        return output.reshape(batch_size, subsampled_time, subsampled_freq * d_model)
    

class FeedForwardBlock(nnx.Module):
    def __init__(self, d_model, expansion_factor, dropout, rngs: nnx.Rngs):
        self.module = nnx.Sequential(
            nnx.LayerNorm(d_model, rngs=rngs),
            nnx.Linear(d_model, d_model * expansion_factor, rngs=rngs),
            nnx.silu,
            nnx.Dropout(rate=dropout, rngs=rngs),
            nnx.Linear(d_model * expansion_factor, d_model, rngs=rngs),
            nnx.Dropout(rate=dropout, rngs=rngs)
        )

    def __call__(self, x):
        return self.module(x)
    

class ConvBlock(nnx.Module):
    def __init__(self, d_model, dropout, rngs: nnx.Rngs):
        self.layer_norm = nnx.LayerNorm(d_model, rngs=rngs)

        self.module = nnx.Sequential(
            nnx.Conv(in_features=d_model, out_features=d_model * 2, kernel_size=1, rngs=rngs),
            nnx.glu,
            nnx.Conv(in_features=d_model, out_features=d_model, kernel_size=31, feature_group_count=d_model, rngs=rngs),
            nnx.BatchNorm(d_model, rngs=rngs),
            nnx.silu,
            nnx.Conv(in_features=d_model, out_features=d_model, kernel_size=1, rngs=rngs),
            nnx.Dropout(rate=dropout, rngs=rngs)
        )

    def __call__(self, x):
        x = self.layer_norm(x)
        x = self.module(x)
        return x
    

class ConformerBlock(nnx.Module):
    def __init__(self,
          d_model=144,
          feed_forward_residual_factor=.5,
          feed_forward_expansion_factor=4,
          num_head=4,
          dropout=0.1,
          training=True,
          rngs: nnx.Rngs = None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.training = training
        self.residual_factor = feed_forward_residual_factor
        self.ff1 = FeedForwardBlock(d_model, feed_forward_expansion_factor, dropout, rngs=rngs)
        self.ln_before_attention = nnx.LayerNorm(d_model, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(num_head, d_model, dropout_rate=dropout, decode=False, rngs=rngs)
        self.conv_block = ConvBlock(d_model, dropout=dropout, rngs=rngs)
        self.ff2 = FeedForwardBlock(d_model, feed_forward_expansion_factor, dropout, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(d_model, rngs=rngs)


    def __call__(self, x, mask=None):
        x = x + (self.residual_factor * self.ff1(x))
        x = x + self.attention(self.ln_before_attention(x), mask=mask, deterministic=not self.training)
        x = x + self.conv_block(x)
        x = x + (self.residual_factor * self.ff2(x))
        return self.layer_norm(x)
    


class ConformerEncoder(nnx.Module):
    def __init__(self, token_count, d_input=80, d_model=144, num_layers=4, feed_forward_residual_factor=0.5, feed_forward_expansion_factor=4, num_head=4, dropout=0.1, rngs: nnx.Rngs = None):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.mel_spectogram = MelSpectrogram(rngs=rngs)
        self.conv_subsampler = Conv2dSubSampler(d_model=d_model, rngs=rngs)
        self.linear_proj = nnx.Linear(d_model * (((d_input - 1) // 2 - 1) // 2), d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

        self.layers = nnx.List([ConformerBlock(d_model=d_model, feed_forward_residual_factor=feed_forward_residual_factor, feed_forward_expansion_factor=feed_forward_expansion_factor,
                                               num_head=num_head, dropout=dropout, rngs=rngs) for _ in range(num_layers)])
        self.decoder = nnx.Linear(d_model, token_count, rngs=rngs)
        

    def __call__(self, x, mask=None, training=True):
        x = self.mel_spectogram(x, training)
        x = self.conv_subsampler(x[:, :, :, None])
        x = self.linear_proj(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.decoder(x)
    

class LSTMDecoder(nnx.Module):
    def __init__(self, d_encoder=144, d_decoder=320, num_layers=1, num_classes=48, rngs: nnx.Rngs = None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        self.d_decoder = d_decoder
        self.num_layers = num_layers
        
        # Create LSTM cells for each layer using nnx.List
        lstm_cells = []
        for i in range(num_layers):
            input_size = d_encoder if i == 0 else d_decoder
            cell = nnx.LSTMCell(
                in_features=input_size,
                hidden_features=d_decoder,
                rngs=rngs
            )
            lstm_cells.append(cell)
        
        self.lstm_cells = nnx.List(lstm_cells)
        
        # Output projection layer
        self.linear = nnx.Linear(d_decoder, num_classes, rngs=rngs)
    
    def __call__(self, x):
        batch_size, time_steps, _ = x.shape
        
        carry = []
        for _ in range(self.num_layers):
            c = jnp.zeros((batch_size, self.d_decoder))
            h = jnp.zeros((batch_size, self.d_decoder))
            carry.append((c, h))
        
        # Process sequence step by step
        outputs = []
        for t in range(time_steps):
            x_t = x[:, t, :]  # (batch_size, d_encoder)
            
            # Pass through each LSTM layer
            for layer_idx in range(self.num_layers):
                c, h = carry[layer_idx]
                carry_new, h_new = self.lstm_cells[layer_idx]((c, h), x_t)
                carry[layer_idx] = carry_new  # carry_new is (c_new, h_new)
                x_t = h_new  # Use hidden state as input to next layer
            
            outputs.append(x_t)
        
        # Stack outputs: (batch_size, time, d_decoder)
        lstm_out = jnp.stack(outputs, axis=1)
        
        # Apply linear projection: (batch_size, time, num_classes)
        logits = nnx.vmap(self.linear, in_axes=1, out_axes=1)(lstm_out)
        
        return logits
    

class ConformerModel(nnx.Module):
    def __init__(self, token_count, rngs: nnx.Rngs = None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.encoder = ConformerEncoder(token_count=token_count, rngs=rngs)
        # self.decoder = LSTMDecoder(rngs=rngs)

    def __call__(self, x, mask, training):
        output = self.encoder(x, mask, training)
        # return self.decoder(output)
        return output