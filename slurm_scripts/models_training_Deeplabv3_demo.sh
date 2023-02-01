#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:1
#SBATCH --error=models/outputs/models_training_Deeplabv3_demo.err
#SBATCH --output=models/outputs/models_training_Deeplabv3_demo.out

python models_training_Deeplabv3_demo.py --evaluate=False --n_epochs=30
