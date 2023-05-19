# Speaker-invariant Clustering (Spin)

 - [Introduction](#Introduction)
 - [Citation](#Citation)  
 - [Getting Started](#Getting-Started)
 - [Pre-trained Models](#Pre-trained-Models)
 - [References](#References)
 - [Contact](#Contact)
     <!-- - [Environment](#1.-Environment)
     - [Prepare Data](#2.-Prepare-Data)
     - [Customize Configurations](#3.-Customize-Configurations)
     - [Training](#4.-Training)
     - [Downstream Evaluation](#5.-Downstream-Evaluation) -->

## Introduction

<p align="center"><img src="https://github.com/vectominist/spin/blob/main/figure/spin.png?raw=true" alt="Spin framework." width="800"/></p>

This repository is the official PyTorch implementation of the **Speaker-invariant Clustering** (**Spin**) proposed in the **Interspeech 2023** paper [Self-supervised Fine-tuning for Improved Content Representations by Speaker-invariant Clustering](https://arxiv.org/abs/2305.11072) ([Heng-Jui Chang](https://people.csail.mit.edu/hengjui/), [Alexander H. Liu](https://alexander-h-liu.github.io/), [James Glass](https://www.csail.mit.edu/person/jim-glass); [MIT CSAIL](https://www.csail.mit.edu/)).

Spin is a novel self-supervised learning method that clusters speech representations and performs swapped prediction between the original and speaker-perturbed utterances. Spin *disentangles speaker information* and preserves *content representations* with just 45 minutes of fine-tuning on a single GPU (HuBERT Base models). Spin improves pre-trained networks and outperforms prior methods in speech recognition and acoustic unit discovery.


## Citation
Please cite our paper if you find this repository and/or the paper useful.
```
@inproceedings{chang2023spin,
  author={Heng-Jui Chang and Alexander H. Liu and James Glass},
  title={{Self-supervised Fine-tuning for Improved Content Representations by Speaker-invariant Clustering}},
  year=2023,
  booktitle={Proc. Interspeech}
}
```


## Getting Started

### 1. Environment
Make sure `sox` is installed and your Python version is at least `3.6`.
```bash
# Create virtual environment
conda create --name spin python=3.8
conda activate spin

# Install s3prl
git clone https://github.com/s3prl/s3prl.git
cd s3prl
pip install -e ".[all]"
cd ..

# Clone this repository and intall dependencies
git clone https://github.com/vectominist/spin.git
cd spin/
pip install -r requirements.txt

# Modify some s3prl files
cp s3prl_py/wav2vec2_model.py ../s3prl/s3prl/upstream/wav2vec2/wav2vec2_model.py
cp s3prl_py/WavLM.py ../s3prl/s3prl/upstream/wavlm/WavLM.py
```


### 2. Prepare Data
Download required data.
```bash
# Create a directory to save data (or any other path you like)
mkdir data
cd data

# LibriSpeech (skip if you already have this)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
wget https://www.openslr.org/resources/12/dev-other.tar.gz
# Decompress
tar zxvf train-clean-100.tar.gz
tar zxvf dev-clean.tar.gz
tar zxvf dev-clean.tar.gz
rm train-clean-100.tar.gz dev-clean.tar.gz dev-clean.tar.gz

# LibriSpeech Phoneme Alignments (for monitoring progress only)
wget https://huggingface.co/datasets/vectominist/spin_data/resolve/main/dev-clean.tsv
wget https://huggingface.co/datasets/vectominist/spin_data/resolve/main/dev-other.tsv

# Speaker Information
# Source: https://github.com/auspicious3000/contentvec
wget https://huggingface.co/datasets/vectominist/spin_data/resolve/main/spk2info.dict
```

Prepare LibriSpeech dataset, see [`script/prepare.sh`](https://github.com/vectominist/spin/blob/main/script/prepare.sh).
- `libri_dir`: the directory of the LibriSpeech corpus
- `json_dir`: the directory to save `.json` files generated from `prepare_data.py`
```bash
bash script/prepare.sh ${libri_dir} ${json_dir}
```

### 3. Customize Configurations
See [`config/spin.yaml`](https://github.com/vectominist/spin/blob/main/config/spin.yaml).
- Modify `json_dir`, `spk2info`, and `phn_dir` according to the directories with the downloaded and preprocessed data.
- Modify `logger` to switch to other loggers or simply setting it to `False` to disable logging.
```yaml
data:
  json_dir: /path/to/json_dir
  spk2info: /path/to/spk2info.dict

val_data:
  json_dir: /path/to/json_dir
  phn_dir: /path/to/phoneme/alignments/dir

trainer:
  logger: wandb  # specify a pytorch-lightning logger you prefer
```

### 4. Training
See [`script/train.sh`](https://github.com/vectominist/spin/blob/main/script/train.sh).
- `exp_dir`: the directory to save checkpoints
- `exp_name`: experiment name
- See [`src/task/train_spin.py`](https://github.com/vectominist/spin/blob/main/src/task/train_spin.py) for details about available arguments like number of GPUs to be used.
```bash
bash script/train.sh ${exp_dir} ${exp_name}
```
The trained model checkpoints can be found in `${exp_dir}/${exp_name}`. Note that we use `last.ckpt` for evaluation and downstream tasks.

### 5. Downstream Evaluation
We use the [s3prl](https://github.com/s3prl/s3prl) toolkit for [SUPERB](https://arxiv.org/abs/2105.01051) downstream tasks.
- Modify [line 26](https://github.com/vectominist/spin/blob/main/s3prl_py/spin/expert.py#L26) of [`s3prl_py/spin/expert.py`](https://github.com/vectominist/spin/blob/main/s3prl_py/spin/expert.py) to the absolute path to `spin/`.
- Copy the `s3prl_py/spin` directory to `s3prl` so that the toolkit can load the models.
  ```bash
  cp -R s3prl_py/spin ../s3prl/s3prl/upstream/spin
  ```
- Finally, add the following line to `../s3prl/s3prl/hub.py`:
  ```python
  from s3prl.upstream.spin.hubconf import *
  ```


## Pre-trained Models
All models are trained on a single NVIDIA A5000 GPU with 24GB VRAM. To reproduce similar or better performance, we suggest using GPUs larger than 24GB or specifying `strategy: ddp` under `trainer` in [`config/spin.yaml`](https://github.com/vectominist/spin/blob/main/config/spin.yaml) to enable multiple GPU training. Note that the following checkpoints are reproduced with the same recipe, so the results are slightly different from our paper. The training logs can be found in this [link](https://api.wandb.ai/links/vectominist/5254la3b).

| Base Model | Clusters | PNMI  | Checkpoint                                                                                       |
| ---------- | -------- | ----- | ------------------------------------------------------------------------------------------------ |
| HuBERT     | 128      | 0.625 | [link](https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_128.ckpt)  |
| HuBERT     | 256      | 0.658 | [link](https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_256.ckpt)  |
| HuBERT     | 512      | 0.707 | [link](https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_512.ckpt)  |
| HuBERT     | 1024     | 0.745 | [link](https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_1024.ckpt) |
| HuBERT     | 2048     | 0.774 | [link](https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_2048.ckpt) |
| WavLM      | 128      | 0.604 | [link](https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_wavlm_128.ckpt)   |
| WavLM      | 256      | 0.658 | [link](https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_wavlm_256.ckpt)   |
| WavLM      | 512      | 0.714 | [link](https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_wavlm_512.ckpt)   |
| WavLM      | 1024     | 0.748 | [link](https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_wavlm_1024.ckpt)  |
| WavLM      | 2048     | 0.775 | [link](https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_wavlm_2048.ckpt)  |


## References
- [s3prl](https://github.com/s3prl/s3prl)
- [contentvec](https://github.com/auspicious3000/contentvec)
- [fairseq](https://github.com/facebookresearch/fairseq)
- [MiniASR](https://github.com/vectominist/MiniASR)
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)


## Contact
If you have any questions, please open an issue or send me an email hengjui@mit.edu.
