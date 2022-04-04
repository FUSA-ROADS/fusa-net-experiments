#!/bin/bash

STAGE_NAME=$1

# Partition and GPU
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:1
 
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

# Identification
#SBATCH -J FUSA-training-%j
#SBATCH -o FUSA-training-%j.out

FUSA_folder="/home/shared/FUSA"

pwd
date
srun --container-name=fusa-torch dvc repro $STAGE_NAME
