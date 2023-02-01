#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --error=models/outputs/models_training_CGNet_demo.err
#SBATCH --output=models/outputs/models_training_CGNet_demo.out

python models_training_demo.py --model_name=cgnet --evaluate=False --n_epochs=30
