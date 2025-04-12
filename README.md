#### S2T model implementation using flax nnx

* implementing [original conformer](https://arxiv.org/pdf/2005.08100)
* using [common voice dataset](https://commonvoice.mozilla.org/en/datasets) and [hugging face](https://github.com/huggingface/datasets) for processing on multi tpu/gpu setup
* finetuning small <100m parameters model on georgian language. Preferably on [kaggle v3-8 TPUS](https://www.kaggle.com/docs/tpu)