#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --error=models/outputs/models_training_Deeplabv3_demo.err
#SBATCH --output=models/outputs/models_training_Deeplabv3_demo.out

python models_training_demo.py --model_name=deeplab --evaluate=True --n_epochs=20
