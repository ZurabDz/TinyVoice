#### S2T model implementation using flax nnx

* implementing [original conformer](https://arxiv.org/pdf/2005.08100)
* using [common voice dataset](https://commonvoice.mozilla.org/en/datasets) and [grain](https://github.com/google/grain) for processing on multi tpu/gpu setup
* finetuning small <100m parameters model on georgian language. Preferably on [kaggle v3-8 TPUS](https://www.kaggle.com/docs/tpu)


## TODOS

- [ ] avoiding train step jit recompilation with fixed shaped inputs
    - [ ] padding each audios to some magic MAX_LENGTH
    - [ ] padding each labels to some magic MAX_LENGTH
    - [ ] explore option to precompile on multiple shapes? at least will have multiple magic MAX_LENGTHES?
- [ ] actually running a training
    - [ ] simply try it out on ASR first
    - [ ] if first attempts do not yeld a result switch towards simpler ASR dataset at first