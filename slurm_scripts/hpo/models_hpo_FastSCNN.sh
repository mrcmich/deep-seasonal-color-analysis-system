#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --error=models/outputs/models_hpo_FastSCNN.err
#SBATCH --output=models/outputs/models_hpo_FastSCNN.out

python models_training_or_hpo.py --config=hpo --model_name=fastscnn --evaluate=True --n_epochs=10
