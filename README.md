# SGDiff: Structure-Guided Diffusion for Cross-Phase Abdominal CT Translation

**Volumetric diffusion modeling for multiphasic abdominal CT synthesis**

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1.2-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-research%20code-orange.svg)](#reproducibility-and-release-checklist)
[![Data](https://img.shields.io/badge/data-available%20on%20request-lightgrey.svg)](#data-availability)

> **Research code.** The in-house MECT dataset is available upon reasonable request, subject to institutional approval and applicable data-use requirements. This README documents the current backbone training entry point and identifies the additional components needed for full paper reproduction.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Availability](#2-data-availability)
3. [Repository Structure](#3-repository-structure)
4. [Requirements](#4-requirements)
5. [Installation](#5-installation)
6. [Dataset Preparation](#6-dataset-preparation)
7. [Pretrained Checkpoints](#7-pretrained-checkpoints)
8. [Quick Start: Train the Verified Diffusion Backbone](#8-quick-start-train-the-verified-diffusion-backbone)
9. [Configuration and Command-Line Arguments](#9-configuration-and-command-line-arguments)
10. [Training Outputs and Checkpoints](#10-training-outputs-and-checkpoints)
11. [Sampling and Inference](#11-sampling-and-inference)
12. [Paper Experiments and Evaluation](#12-paper-experiments-and-evaluation)
13. [Troubleshooting](#13-troubleshooting)
14. [Reproducibility and Release Checklist](#14-reproducibility-and-release-checklist)
15. [Citation and License](#15-citation-and-license)

---

## 1. Overview

**SGDiff** (Structure-Guided Diffusion) studies the synthesis of missing contrast-enhancement phases in abdominal computed tomography (CT). The framework combines volumetric latent diffusion with anatomical guidance to model phase-dependent appearance while preserving patient-specific structures.

The research methodology comprises:

- A **3D latent diffusion backbone** with interleaved spatial and inter-slice Transformer blocks.
- **Pretrained visual representations** to support anatomical and semantic modeling.
- A **Frequency Mask Predictor (FMP)** to obtain target-phase-oriented high-frequency structural cues.
- **ControlNet-Trans** to incorporate structural conditions into diffusion denoising.
- A **Target-Similarity Loss (TSL)** to regularize local structural relationships.

### ⚠️ Important distinction

The verified entry point `train_scripts/train_sgdiff.py` trains a **class-conditional, volumetric latent diffusion backbone**. Running this file alone is **not** equivalent to training or evaluating the complete FMP + ControlNet-Trans + TSL cross-phase translation pipeline. This README does not invent unverified commands for those additional components.

| Component | Status in this repository |
| :--- | :--- |
| 3D latent diffusion backbone training | Verified entry point: `train_scripts/train_sgdiff.py` |
| 3D VAE (AutoencoderKL) architecture and config | Present |
| FMP, ControlNet-Trans, TSL | Described in the paper; not covered by the verified entry point |
| In-house MECT dataset | Available on request (see below) |
| VinDr-Multiphase experiments | Separate dataset, obtained from its official provider |

---

## 2. Data Availability

### 2.1 In-house MECT dataset — available upon request

The in-house **MECT** dataset comprises multiphasic abdominal contrast-enhanced CT studies used in the SGDiff research. It is **not** hosted in this GitHub repository and is **not** available as a public direct download because it contains clinical imaging data.

**Data Availability Statement.** The in-house MECT dataset is available upon reasonable request to the corresponding author, subject to institutional approval, applicable privacy and ethical requirements, and execution of any necessary data-use agreements. Requests will be evaluated for their scientific purpose and compliance with relevant data-sharing policies.

**How to request access.** Contact the corresponding author of the associated SGDiff paper and provide:

- your institutional affiliation,
- the intended research use,
- the requested data scope,
- a brief research proposal.

> The maintainers should add a direct contact email to this section once an authorized address is available. Submission of a request does not guarantee approval or immediate transfer.

- The manuscript describes an in-house MECT cohort of **622 patients** with arterial (**ART**), portal venous (**PV**), and delayed (**DL**) CT phases.
- Data provided under an approved request may be subject to de-identification, secure transfer, use restrictions, and institutional review.
- Do **not** open a public GitHub issue containing patient information, credentials, or other sensitive material.

### 2.2 VinDr-Multiphase

[VinDr-Multiphase](https://vindr.ai/) is a separate dataset used for the NC-to-ART and cascaded ART-to-PV experiments. Obtain it independently from the dataset's official provider and comply with its access terms. The MECT-specific loader documented below does **not** automatically support the VinDr non-contrast (NC) task.

---

## 3. Repository Structure

The following is the target layout after the one-time MECT naming synchronization (selected files, not an exhaustive listing). The upstream GitHub checkout may still contain legacy filenames until the changes in [Section 5.1](#51-one-time-mect-naming-synchronization) are applied.

```text
SGDiff/
├── Exp_Classify/                  # Downstream classification experiments
├── autoencoder/
│   ├── vae.py                     # 3D AutoencoderKL
│   ├── modules/
│   │   └── aemodule3D.py          # 3D encoder / decoder implementations
│   └── losses/                    # VAE reconstruction / perceptual losses
├── config/
│   ├── sgdiff/
│   │   └── train/
│   │       └── mect_v1.yaml       # MECT diffusion training configuration
│   └── vae/
│       └── pretrain_vae.yaml      # Verified VAE architecture configuration
├── datas/
│   ├── __init__.py                # MONAI transforms / dataset factory
│   ├── mect_dataset.py            # MECT NIfTI loader
│   └── data_util.py               # Data utilities and Sobel operator
├── diffusion/
│   └── __init__.py                # Diffusion construction
├── model/
│   ├── latte/
│   │   ├── __init__.py            # Model selection
│   │   └── latte.py               # Transformer diffusion backbone
│   └── dino/
│       └── vision_transformer.py  # DINO visual encoder
├── sample/                        # Sampling-related code; see notes below
├── train_scripts/
│   ├── train_sgdiff.py            # Verified training entry point
│   └── train_util.py              # Model loading, EMA, logging, etc.
├── dino_test.py                   # Standalone DINO loading example
└── utils.py
```

---

## 4. Requirements

| Item | Requirement |
| :--- | :--- |
| OS | Linux recommended |
| GPU | NVIDIA GPU with a compatible CUDA driver (required by the verified training script) |
| Python | 3.10 (recommended starting point) |
| Memory | Sufficient GPU memory for a 3D, 28-block Transformer — start with `--local_batch_size 1` if memory is limited |
| Data | Approved access to the in-house MECT NIfTI dataset (or suitably prepared data with the same structure) |
| Weights | A compatible pretrained 3D VAE checkpoint |

The source imports PyTorch, torchvision, MONAI, PyTorch Lightning, timm, einops, omegaconf, diffusers, OpenCV, imageio, Pillow, matplotlib, and other utilities.

> **Note:** There is currently no verified root-level `requirements.txt` or lock file. The installation recipe below is a **suggested baseline**, not a tested, version-locked reproduction environment.

---

## 5. Installation

### 5.0 Clone the repository

```bash
git clone https://github.com/lzhLab/SGDiff.git
cd SGDiff
```

For the current public checkout, complete the one-time MECT naming synchronization below **before** using the training commands.

### 5.1 One-time MECT naming synchronization

All active dataset settings in this README use **MECT**. At the time of documentation review, the public GitHub checkout still uses legacy `MPECT` identifiers in its Python dataset factory, YAML settings, and some filenames. A README edit alone does not change what Python accepts.

From the repository root, run:

```bash
python - <<'PY'
from pathlib import Path

renames = {
    Path("datas/mpect_dataset.py"): Path("datas/mect_dataset.py"),
    Path("config/sgdiff/train/mpect_v1.yaml"): Path("config/sgdiff/train/mect_v1.yaml"),
}
for old, new in renames.items():
    if old.exists() and not new.exists():
        old.rename(new)
        print(f"Renamed {old} -> {new}")
    elif new.exists():
        print(f"Already present: {new}")
    else:
        raise FileNotFoundError(f"Missing source and target: {old}, {new}")

edits = {
    Path("datas/__init__.py"): {
        "mpect_dataset": "mect_dataset",
        "mpect_transforms": "mect_transforms",
        "'MPECT'": "'MECT'",
    },
    Path("config/sgdiff/train/mect_v1.yaml"): {
        "dataset: MPECT": "dataset: MECT",
    },
    Path("train_scripts/train_sgdiff.py"): {
        "mpect_v1.yaml": "mect_v1.yaml",
        "/MPECT": "/MECT",
    },
    Path("datas/mect_dataset.py"): {
        "/MPECT": "/MECT",
    },
}
for file, replacements in edits.items():
    source = file.read_text(encoding="utf-8")
    updated = source
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    if updated != source:
        file.write_text(updated, encoding="utf-8")
        print(f"Updated {file}")

print("MECT naming synchronized. Review and commit the changes.")
PY
```

Confirm the active dataset identifier:

```bash
grep -n '^dataset: MECT$' config/sgdiff/train/mect_v1.yaml
grep -n "args.dataset == 'MECT'" datas/__init__.py
```

Audit any remaining legacy identifiers in source/config files:

```bash
git grep -n -E 'MPECT|mpect' -- '*.py' '*.yaml' '*.yml' '*.sh' || true
```

> These commands change files in your **local clone only**. They do not update the remote GitHub repository. Review the diff and commit/push the changes together with this README. Any additional legacy matches found by the audit should be updated before claiming repository-wide naming consistency.

### 5.2 Create an environment

```bash
conda create -n sgdiff python=3.10 -y
conda activate sgdiff

python -m pip install --upgrade pip setuptools wheel
```

### 5.3 Install PyTorch

Install a PyTorch / torchvision pair compatible with your GPU driver. For example, if your driver supports the CUDA 12.1 PyTorch binaries, one possible baseline is:

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions, use the command provided by the official [PyTorch installation selector](https://pytorch.org/get-started/locally/). Do not install a CUDA build incompatible with your driver.

### 5.4 Install the remaining dependencies

The following is a suggested baseline derived from the source imports; pin versions as needed for your environment:

```bash
python -m pip install \
  monai pytorch-lightning timm einops omegaconf diffusers \
  opencv-python imageio pillow matplotlib numpy scipy
```

---

## 6. Dataset Preparation

The MECT loader expects NIfTI volumes organized under the directory declared in `config/sgdiff/train/mect_v1.yaml`. Before training, make sure that:

1. Your approved MECT data has been de-identified and transferred according to your data-use agreement.
2. Each study is stored as a NIfTI volume (`.nii` / `.nii.gz`) following the layout expected by `datas/mect_dataset.py`.
3. The `data_root` (or equivalent path entry) in `config/sgdiff/train/mect_v1.yaml` points to your local copy.
4. The phase labels (ART / PV / DL) are consistent with the configuration.

> The MECT loader does not support the VinDr non-contrast (NC) task out of the box; the NC-to-ART experiments require separate preparation.

---

## 7. Pretrained Checkpoints

Training the diffusion backbone requires a compatible **pretrained 3D VAE checkpoint** (3D `AutoencoderKL`). The VAE architecture is configured in `config/vae/pretrain_vae.yaml`.

Checkpoints are not distributed in this repository. Place your VAE (and, if applicable, DINO visual encoder) weights in a local directory and reference that path in the training configuration or command line.

---

## 8. Quick Start: Train the Verified Diffusion Backbone

With the environment prepared, the MECT data in place, and the VAE checkpoint available:

```bash
python train_scripts/train_sgdiff.py \
  --config config/sgdiff/train/mect_v1.yaml \
  --local_batch_size 1
```

> The exact argument names and defaults are defined in `train_scripts/train_sgdiff.py` and the YAML configuration. Inspect both before launching a long run, and verify the resolved config path, dataset root, and checkpoint path.

---

## 9. Configuration and Command-Line Arguments

Configuration is split between the YAML file and CLI overrides:

- `config/sgdiff/train/mect_v1.yaml` — dataset identifier, data root, model/Transformer depth, diffusion schedule, optimizer, logging, and checkpoint settings.
- `train_scripts/train_sgdiff.py` — command-line entry point and argument parsing.
- `train_scripts/train_util.py` — model loading, EMA, and logging utilities.

Recommended practice when adapting the recipe:

1. Copy `mect_v1.yaml` to a new file for your experiment rather than editing in place.
2. Keep `--local_batch_size 1` for the first run to confirm the pipeline fits in GPU memory.
3. Re-check `dataset`, data root, and VAE checkpoint path after any rename or relocation.

---

## 10. Training Outputs and Checkpoints

Training writes logs, checkpoints, and EMA weights to the output directory configured in the YAML. Expect the following artifacts:

- periodic model checkpoints,
- EMA weights used for sampling,
- training logs (loss curves and scalar metrics).

> Exact directory names and file naming follow the settings in `mect_v1.yaml` and the logging utilities in `train_scripts/train_util.py`.

---

## 11. Sampling and Inference

Sampling-related code lives in `sample/`. Because the full cross-phase translation pipeline (FMP + ControlNet-Trans + TSL) is not part of the verified entry point, this README does not provide an end-to-end inference command. To sample from the trained backbone, use the sampling utilities in `sample/` together with the EMA checkpoint produced by training, and adapt the script to your checkpoint layout.

> Any sampling script you add should be validated on a small case before being used for quantitative evaluation.

---

## 12. Paper Experiments and Evaluation

The paper reports results on:

- **In-house MECT** — arterial (ART), portal venous (PV), and delayed (DL) phases, 622 patients.
- **VinDr-Multiphase** — NC-to-ART and cascaded ART-to-PV experiments, using the dataset obtained independently from its official provider.

Full reproduction requires components beyond the verified backbone entry point. See [Section 1](#1-overview) for the explicit boundary between what this repository trains and what the paper additionally describes.

---

## 13. Troubleshooting

| Symptom | Likely cause / action |
| :--- | :--- |
| `ModuleNotFoundError: datas.mpect_dataset` | The one-time MECT naming synchronization ([5.1](#51-one-time-mect-naming-synchronization)) has not been applied. |
| Dataset not found / empty dataloader | Check the data root in `mect_v1.yaml` and the NIfTI layout expected by `datas/mect_dataset.py`. |
| CUDA out of memory | Use `--local_batch_size 1`, reduce volume size/crop in the config, or use a GPU with more memory. |
| CUDA driver / binary mismatch | Reinstall the PyTorch build matching your driver from the official selector. |
| Missing VAE checkpoint | Provide a compatible pretrained 3D VAE and point the config to it. |

---

## 14. Reproducibility and Release Checklist

- [ ] Clone the repository and apply the one-time MECT naming synchronization.
- [ ] Confirm `dataset: MECT` in `config/sgdiff/train/mect_v1.yaml` and `args.dataset == 'MECT'` in `datas/__init__.py`.
- [ ] Audit remaining legacy `MPECT` identifiers with `git grep`.
- [ ] Obtain approved access to the MECT dataset and place it locally.
- [ ] Obtain a compatible pretrained 3D VAE checkpoint.
- [ ] Record your exact environment (driver, CUDA, PyTorch, MONAI, timm versions) — no lock file is provided.
- [ ] Run a short smoke test with `--local_batch_size 1` before full training.
- [ ] Note explicitly that the FMP + ControlNet-Trans + TSL pipeline is not covered by the verified entry point.

---

## 15. Citation and License

If you use this code or the MECT dataset in your research, please cite the associated SGDiff paper.

```bibtex
@article{sgdiff,
  title   = {SGDiff: Structure-Guided Diffusion for Cross-Phase Abdominal CT Translation},
  author  = {Yiqi Wang, Qiang He, Qiuhao Zong, Jin Zhang, Zhicai Peng, Menglin Zhao, Liang Zhao},
  journal = {arxiv.org},
  year    = {2026}
}
```

> Replace the placeholder fields above with the final bibliographic details before release.

**License.** No license file is currently included in this repository. Add one (for example, an academic/research-only license) before distributing the code or accepting external contributions.

---

<p align="center"><sub>SGDiff · research code · data available upon reasonable request</sub></p>
