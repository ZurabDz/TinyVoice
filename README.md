#### S2T model implementation using flax nnx

* implementing [original conformer](https://arxiv.org/pdf/2005.08100)
* using [common voice dataset](https://commonvoice.mozilla.org/en/datasets) and [grain](https://github.com/google/grain) for processing on multi tpu/gpu setup
* finetuning small <100m parameters model on georgian language. Preferably on [kaggle v3-8 TPUS](https://www.kaggle.com/docs/tpu)


## TODOS

- [ ] avoiding train step jit recompilation with fixed shaped inputs
    - [x] padding each audios to some magic MAX_LENGTH
    - [x] padding each labels to some magic MAX_LENGTH
    - [ ] explore option to precompile on multiple shapes? at least will have multiple magic MAX_LENGTHES?
        - [ ] explore [EasyDel](https://github.com/erfanzar/EasyDeL) multi compile step options
- [ ] actually running a training
    - [x] simply try it out on ASR first


#### First run on google colab tpu v5e-1 looked pretty good, some bugs need to be ironed out

- [ ] Training Fixes
    - [ ] mixed precision is weird, som norm are fp32 some bf16
    - [ ] detokenizer is weird taking list[str] instead of list[int]
    - [ ] tpu req attention fp32 and doesn't work on bf16? what? probably my mystake cause on rtx 2070 Super it does work
    - [ ] 80% tpu util 5GB "VRAM" usage with 64 batch. What? will look into it. rtx 2070 took 8Gb with 24 batch
    - [ ] config tweaking, model size, batch size, training args
    - [ ] fully working multiple epochs training
    - [ ] model saving and loading + training data saving and loading
    - [ ] check steps and lr schedule do indeed work with optimizer 
    - [ ] need valid and test sets with jitted steps
    - [ ] instead of train.tsv we should combined validated + other tsvs


##### Generic steps

I should have like steps:

* script processing common voice
    * resample audios to 16khz
    * create new train/test split .csv files and add resampled full path + audio durations

* training* tokenizer 
    * loads train data and builds tokenizer. 
    * save tokenizer to given path

* training ipynb