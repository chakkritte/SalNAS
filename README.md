
# SalNAS: Efficient Saliency-prediction Neural Architecture Search with self-knowledge distillation

This paper has been accepted to Engineering Applications of Artificial Intelligence.

Paper: [EAAI version](https://doi.org/10.1016/j.engappai.2024.109030) or [arXiv version](https://arxiv.org/pdf/2407.20062)

![](https://img.shields.io/badge/-PyTorch%20Implementation-blue.svg?logo=pytorch)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## Overview

This repository contains the source code for SalNAS, which accompanies the research paper titled **SalNAS: Saliency-Prediction Neural Architecture Search with Self-Knowledge Distillation**. The purpose of this repository is to provide transparency and reproducibility of the research results presented in the paper.

**This code is based on the implementation of  [EML-NET-Saliency](https://github.com/SenJia/EML-NET-Saliency), [SimpleNet](https://github.com/samyak0210/saliency), [MSI-Net](https://github.com/alexanderkroner/saliency), [EEEA-Net](https://github.com/chakkritte/EEEA-Net), and [AlphaNet](https://github.com/facebookresearch/AlphaNet).**

## Prerequisite for server
 - Tested on Ubuntu OS version 22.04 LTS
 - Tested on Python 3.11.8
 - Tested on CUDA 12.3
 - Tested on PyTorch 2.2.1 and TorchVision 0.17.1
 - Tested on NVIDIA RTX 4090 24 GB

### Cloning source code

```
git clone https://github.com/chakkritte/SalNAS/
cd PKD
mkdir data
```

## The dataset folder structure:

```
PKD
|__ data
    |_ salicon
      |_ fixations
      |_ saliency
      |_ stimuli
    |_ mit1003
      |_ fixations
      |_ saliency
      |_ stimuli
    |_ cat2000
      |_ fixations
      |_ saliency
      |_ stimuli
    |_ pascals
      |_ fixations
      |_ saliency
      |_ stimuli
    |_ osie
      |_ fixations
      |_ saliency
      |_ stimuli
```

### Creating new environments

```
conda create -n salnas python=3.11.8
conda activate salnas
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Install Requirements

```
pip install -r requirements.txt --no-cache-dir
```

## Usage

### Training SalNAS supernet on Salicon dataset
```
python train_salnas.py --amp --self_kd --no_epochs 20 --t_epochs 10  --loss_mode new --kldiv --cc --nss --output_dir output-selfkd
```

## Citation

If you use SalNAS or any part of this research, please cite our paper:
```
@article{termritthikun2024salnas,
  title = "{SalNAS: Efficient Saliency-prediction Neural Architecture Search with self-knowledge distillation}",
  journal = {Engineering Applications of Artificial Intelligence},
  volume = {136},
  pages = {109030},
  year = {2024},
  issn = {0952-1976},
  doi = {https://doi.org/10.1016/j.engappai.2024.109030},
  author = {Chakkrit Termritthikun and Ayaz Umer and Suwichaya Suwanwimolkul and Feng Xia and Ivan Lee},
}
``````


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
