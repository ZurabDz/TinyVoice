# TinyVoice: Hybrid FastConformer ASR with Flax NNX

This project is a high-performance Speech-to-Text (S2T) implementation of the [FastConformer](https://arxiv.org/abs/2305.05084) architecture using [Flax NNX](https://github.com/google/flax) and [JAX](https://github.com/google/jax). It is optimized for multi-TPU/GPU training and focuses on fine-tuning for the Georgian language.

## Project Overview

- **Architecture:** Hybrid CTC/Attention FastConformer.
  - **Encoder:** 16-layer FastConformer with 4x subsampling, RoPE, and Flash Attention.
  - **Decoder:** 4-layer Transformer Decoder for auxiliary Attention-based loss.
  - **Regularization:** Stochastic Depth (DropPath) and Intermediate CTC Loss (at Layer 8).
- **Data Pipeline:** Powered by [Google Grain](https://github.com/google/grain). Currently uses fixed-shape padding (11s).
- **Training:** Hybrid Multi-Objective Loss: `(1-α) * CTC + α * AED + 0.3 * InterCTC`.
- **Optimization:** AdamW with Cosine Annealing and 15k-step linear warmup.

## Directory Structure

- `conformer/`: Core model implementation.
  - `model.py`: Hybrid FastConformer, Encoder, Decoder, and DropPath modules.
  - `config.py`: TrainingArguments (d_model, layers, weights, etc.).
  - `dataset.py`: Grain-based data loading and augmentations.
  - `factory.py`: Model building and checkpoint restoration.
- `scripts/`:
  - `run_training.sh`: Main entry point (train, dev, profile, debug).
  - `process_common_voice_dataset.py`: Dataset preparation.
- `train_minimal.py`: Main training loop with hybrid loss and teacher forcing.

## Building and Running

### Setup
```bash
uv sync
```

### Training
```bash
# Start full hybrid training
./scripts/run_training.sh train

# Run a quick development test
./scripts/run_training.sh dev
```

### Inference
```bash
# Run evaluation on test set
uv run python inference.py

# Transcribe a single file
uv run python inference_single.py path/to/audio.wav
```

## Convergence & Performance Tips
- **Intermediate CTC:** The `interctc_weight` (default 0.3) is critical for early convergence.
- **Attention Weight:** `attention_weight` (default 0.2) helps the model learn Georgian language structures.
- **Warmup:** Do not skip the 15k-step warmup; it stabilizes the hybrid gradients.
