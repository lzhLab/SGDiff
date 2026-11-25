#!/bin/bash
#SBATCH -J SGDiff
#SBATCH -o SGDiff-%j.log
#SBATCH -e SGDiff-%j.err
#SBATCH -N 1
#SBATCH --gres=gpu:1
source activate lzp_env

python train_sgdiff.py


