#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --error=models/outputs/tuned_UNet.err
#SBATCH --output=models/outputs/tuned_UNet.out

python models_tuned_UNet.py
