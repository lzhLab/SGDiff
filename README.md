SGDiff: Structure-Guided Diffusion for Cross-Phase Abdominal CT Translation

Status: Research code. The in-house MECT dataset is available upon reasonable request, subject to institutional approval and applicable data-use requirements. This README documents the current backbone training entry point and identifies additional components needed for full paper reproduction.
Overview
SGDiff (Structure-Guided Diffusion) studies the synthesis of missing contrast-enhancement phases in abdominal computed tomography (CT). The research framework combines volumetric latent diffusion with anatomical guidance to model phase-dependent appearance while preserving patient-specific structures.
The research methodology describes:
- A 3D latent diffusion backbone with interleaved spatial and inter-slice Transformer blocks.
- Pretrained visual representations to support anatomical and semantic modeling.
- A Frequency Mask Predictor (FMP) to obtain target-phase-oriented high-frequency structural cues.
- ControlNet-Trans to incorporate structural conditions into diffusion denoising.
- A Target-Similarity Loss (TSL) to regularize local structural relationships.
Important distinction: The verified entry point train_scripts/train_sgdiff.py trains a class-conditional, volumetric latent diffusion backbone. Running this file alone is not equivalent to training or evaluating the complete FMP + ControlNet-Trans + TSL cross-phase translation pipeline. The README does not invent unverified commands for those additional components.
Data availability
In-house MECT dataset — available upon request
The in-house MECT dataset comprises multiphasic abdominal contrast-enhanced CT studies used in the SGDiff research. It is not hosted in this GitHub repository and is not available as a public direct download because it contains clinical imaging data.
Data Availability Statement
The in-house MECT dataset is available upon reasonable request to the corresponding author, subject to institutional approval, applicable privacy and ethical requirements, and execution of any necessary data-use agreements. Requests will be evaluated for their scientific purpose and compliance with relevant data-sharing policies.
How to request access: Contact the corresponding author of the associated SGDiff paper and provide your institutional affiliation, intended research use, requested data scope, and a brief research proposal. The maintainers should add a direct contact email to this section once an authorized address is available. Submission of a request does not guarantee approval or immediate transfer.
- The manuscript describes an in-house MECT cohort of 622 patients with arterial (ART), portal venous (PV), and delayed (DL) CT phases.
- Data provided under an approved request may be subject to de-identification, secure transfer, use restrictions, and institutional review.
- Do not open a public GitHub issue containing patient information, credentials, or other sensitive material.
VinDr-Multiphase
VinDr-Multiphase is a separate dataset used for the NC-to-ART and cascaded ART-to-PV experiments. Obtain it independently from the dataset's official provider and comply with its access terms. The MECT-specific loader documented below does not automatically support the VinDr non-contrast (NC) task.
Contents
1. Data availability
2. Repository structure
3. Requirements
4. Installation
5. Dataset preparation
6. Pretrained checkpoints
7. Quick start: train the verified diffusion backbone
8. Configuration and command-line arguments
9. Training outputs and checkpoints
10. Sampling and inference
11. Paper experiments and evaluation
12. Troubleshooting
13. Reproducibility and release checklist
14. Citation and license
Repository structure
The following is the target layout after the one-time MECT naming synchronization (selected files, not an exhaustive listing). The upstream GitHub checkout may still contain legacy filenames until the changes below are applied:
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
│   │       └── mect_v1.yaml      # MECT diffusion training configuration
│   └── vae/
│       └── pretrain_vae.yaml      # Verified VAE architecture configuration
├── datas/
│   ├── __init__.py                # MONAI transforms / dataset factory
│   ├── mect_dataset.py           # MECT NIfTI loader
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
One-time MECT naming synchronization for the current code
All active dataset settings in this README use MECT. At the time of documentation review, the public GitHub checkout still uses legacy MPECT identifiers in its Python dataset factory, YAML settings, and some filenames. A README edit alone does not change what Python accepts.
Before running the MECT training examples, apply this one-time local migration from the repository root (after git clone). It renames the loader/config files and updates the corresponding code references:
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
Confirm the active dataset identifier:
grep -n '^dataset: MECT$' config/sgdiff/train/mect_v1.yaml
grep -n "args.dataset == 'MECT'" datas/__init__.py

# Audit any remaining legacy identifiers in source/config files:
git grep -n -E 'MPECT|mpect' -- '*.py' '*.yaml' '*.yml' '*.sh' || true
These commands change files in your local clone only. They do not update the remote GitHub repository. Review the diff and commit/push the changes together with this README. Any additional legacy matches found by the audit should be updated before claiming repository-wide naming consistency.
Requirements
- Linux is recommended.
- An NVIDIA GPU with a compatible CUDA driver is required by the verified training script.
- Python 3.10 is a recommended starting point.
- Sufficient GPU memory for a 3D, 28-block Transformer. Start with --local_batch_size 1 if memory is limited.
- Approved access to the in-house MECT NIfTI dataset (or suitably prepared data with the same structure), plus a compatible pretrained 3D VAE checkpoint.
The source imports PyTorch, torchvision, MONAI, PyTorch Lightning, timm, einops, omegaconf, diffusers, OpenCV, imageio, Pillow, matplotlib, and other utilities.
There is currently no verified root-level requirements.txt or lock file. The installation recipe below is a suggested baseline, not a tested, version-locked reproduction environment.
Installation
1. Clone the repository
git clone https://github.com/lzhLab/SGDiff.git
cd SGDiff
For the current public checkout, first complete the one-time MECT naming synchronization before using the training commands below.
2. Create an environment
conda create -n sgdiff python=3.10 -y
conda activate sgdiff

python -m pip install --upgrade pip setuptools wheel
3. Install PyTorch
Install a PyTorch / torchvision pair compatible with your GPU driver. For example, if your driver supports the CUDA 12.1 PyTorch binaries, one possible baseline is:
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121
For other CUDA versions, use the command provided by the official PyTorch installation selector. Do not install a CUDA build incompatible with your driver.
