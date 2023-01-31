#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --error=models/outputs/models_hpo_UNet.err
#SBATCH --output=models/outputs/models_hpo_UNet.out

python models_hpo.py unet