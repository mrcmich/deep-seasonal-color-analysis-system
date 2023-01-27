#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --error=models/outputs/HPO_FastSCNN.err
#SBATCH --output=models/outputs/HPO_FastSCNN.out

python HPO_FastSCNN.py
