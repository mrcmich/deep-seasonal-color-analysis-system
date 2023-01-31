#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --error=models/outputs/models_tuning_FastSCNN.err
#SBATCH --output=models/outputs/models_tuning_FastSCNN.out

python models_tuning_FastSCNN.py
