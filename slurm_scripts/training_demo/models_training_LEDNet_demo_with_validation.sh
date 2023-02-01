#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --error=models/outputs/models_training_LEDNet_demo_with_validation.err
#SBATCH --output=models/outputs/models_training_LEDNet_demo_with_validation.out

python models_training_demo.py --model_name=lednet --evaluate=True --n_epochs=30
