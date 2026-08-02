#### S2T model implementation using flax nnx

* implementing [original conformer](https://arxiv.org/pdf/2005.08100)
* using [common voice dataset](https://commonvoice.mozilla.org/en/datasets) and [grain](https://github.com/google/grain) for processing on multi tpu/gpu setup
* finetuning small <100m parameters model on georgian language. Preferably on [kaggle v3-8 TPUS](https://www.kaggle.com/docs/tpu)

### Dataset preparation

The preparation scripts use a dataset root containing `train.tsv`, `dev.tsv`,
`test.tsv`, and `clips/`. Processed TSVs remain in that root; ArrayRecords and
the tokenizer are written to `packed_dataset`, which is the default training
data directory.

```bash
cd TinyVoice/scripts
./resample_and_prepare.sh /path/to/common_voice_ka
./generate_array_records.sh /path/to/common_voice_ka
python diagnose_dataset.py --require-disjoint-speakers /path/to/common_voice_ka/*_processed.tsv
```

The diagnostic command must show zero duplicate files and, for a
speaker-disjoint evaluation, zero shared `client_id` values between splits.
Do not use personal recordings in the training TSV; keep them as a separate
held-out evaluation set.

### Inference and NPU deployment

See **[INFERENCE.md](INFERENCE.md)** for the full pipeline: exporting a
checkpoint to ONNX, transcribing on the host, converting to an Allwinner NPU
network binary (`.nb`), and building the on-device C runner in `device/`.

Verified end to end on a Radxa Cubie A7S (Allwinner A733): **45x realtime
single-stream, 62x with two workers**, with transcriptions identical to the
float model.

On the workstation, with the `ubuntu-npu` SDK container running:

```bash
python scripts/export_onnx.py --seconds 11 --calib-samples 64   # checkpoint -> ONNX
scripts/npu_convert.sh <container-id> a733 int16                # ONNX -> quantised .nb
python scripts/export_device_assets.py --qtype int16            # filterbank, vocab, header
python scripts/npu_infer.py sample.wav                          # transcribe on the host
```

Then on the board, which builds natively (no cross toolchain):

```bash
make -C device AI_SDK_ROOT=~/ai-sdk AI_SDK_PLATFORM=a733 MODEL_DIR=<device assets>
device/tinyvoice_run model.nb filterbank.bin vocab.txt audio.wav
```

Pass **your** chip to both `npu_convert.sh` and `AI_SDK_PLATFORM` -- the
hardware target is compiled into the `.nb` and the driver rejects a mismatch.
Supported: `a733 ai985 mr527 mr536 t527 t536 t736`.
