#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --error=models/outputs/models_tuning_UNet.err
#SBATCH --output=models/outputs/models_tuning_UNet.out

python models_tuning.py unet
