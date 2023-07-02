# Fine-Grained RLHF

## Note: this repo is still under construction, so many components have not been ready yet! We will launch a initial version in late June.

This repository contains the code for the paper [Fine-Grained Human Feedback Gives Better Rewards for Language Model Training](https://arxiv.org/abs/2306.01693).

## Installation

```bash
pip install -r requirements.txt
```

## Usage
Customize rewards and evaluation metrics for each task in `reward.py`

Dataset is customized in `train_baseline.py` and `train_finegrained.py`

Specify reward model path in yml files

## Run

```bash
bash train_finegrained.sh
bash train_baseline.sh
```

## Citation

```bibtex
@article{wu2023finegrained,
title={Fine-Grained Human Feedback Gives Better Rewards for Language Model Training},
author={Zeqiu Wu and Yushi Hu and Weijia Shi and Nouha Dziri and Alane Suhr and Prithviraj Ammanabrolu and Noah A. Smith and Mari Ostendorf and Hannaneh Hajishirzi},
journal={arXiv preprint arXiv:2306.01693},
year={2023},
url={https://arxiv.org/abs/2306.01693},
}
```