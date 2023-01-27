#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --error=models/outputs/HPO_UNet.err
#SBATCH --output=models/outputs/HPO_UNet.out

python HPO_UNet.py
