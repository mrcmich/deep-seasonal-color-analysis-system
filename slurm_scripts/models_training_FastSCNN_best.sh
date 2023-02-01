#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --error=models/outputs/models_training_FastSCNN_best.err
#SBATCH --output=models/outputs/models_training_FastSCNN_best.out

python models_training_best.py --model_name=fastscnn --evaluate=False
