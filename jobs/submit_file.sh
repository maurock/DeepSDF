#!/bin/bash
#SBATCH --job-name=name
#SBATCH --nodes=1
#SBATCH --partition=cnu
#SBATCH --mem=24G
#SBATCH --exclude=bp1-gpu004,bp1-gpu016,bp1-gpu017
#SBATCH --gpus=1
#SBATCH --time=0-03:30:00
#SBATCH --output=%j.out

cd "/user/work/ri21540/code/DeepSDF/"

python3 train.py