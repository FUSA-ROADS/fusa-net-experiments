#!/bin/bash

# Partition and GPU
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:1 

# Identification
#SBATCH -J FUSA-dataset-test-%j
#SBATCH -o slurm-%j.out

FUSA_folder="/home/shared/FUSA"

pwd
date
mkdir -p logs
srun --container-name=pytorch --container-mounts=$FUSA_folder/datasets:/datasets python datasets.py
mv slurm-*.out logs
