#!/bin/bash

# Partition and GPU
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:2

#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

# Identification
#SBATCH -J fusa-training
#SBATCH -o fusa-training-%j.out
#SBATCH -e fusa-training-%j.err

pwd
date
srun --container-name=fusa-torch dvc repro train -v -f
