#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --error=models/outputs/tuned_FastSCNN.err
#SBATCH --output=models/outputs/tuned_FastSCNN.out

python models_tuned_FastSCNN.py
