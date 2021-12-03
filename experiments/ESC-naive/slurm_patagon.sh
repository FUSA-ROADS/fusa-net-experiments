#!/bin/bash

# Partition and GPU
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:1 

# Identification
#SBATCH -J FUSA-training-%j
#SBATCH -o FUSA-training-%j.out

FUSA_folder="/home/shared/FUSA"

pwd
date
srun --container-name=fusa-torch dvc repro
