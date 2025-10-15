# Language Modeling is Compression

This repository provides an re-implementation of Google Deepmind's ICLR 2024 paper [Language Modeling is Compression](https://arxiv.org/abs/2309.10668).
In the original implementation released by Google, the Haiku framework (paired with JAX) is used—a setup that can present a steep learning curve for beginners, particularly those less familiar with functional programming paradigms. 

To address this accessibility gap, we have reimplemented key components of the paper’s methods using standard PyTorch code (adopting a non-functional programming style). 

Critically, our reimplementation retains the ability to reproduce the exact results reported in the original paper.


## Installation

Pls follow the installation guide in the orginal repo: [language_modeling_is_compression](https://github.com/google-deepmind/language_modeling_is_compression)

## Usage

If you want to compress with a language model, you need to train it first using:
```bash
python train_enwik_torch.py -e 3 -b 128;
```

To evaluate the compression rates, use (assume the model generated in train step is saved as `trained_model.pth`):
```bash
python compress_enwik_torch.py -m trained_model.pth 
```
