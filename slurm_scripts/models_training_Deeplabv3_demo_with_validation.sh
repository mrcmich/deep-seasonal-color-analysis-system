#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:1
#SBATCH --error=models/outputs/models_training_Deeplabv3_demo_with_validation.err
#SBATCH --output=models/outputs/models_training_Deeplabv3_demo_with_validation.out
#SBATCH --signal=USR1@600

srun python models_training_Deeplabv3_demo.py --evaluate=True
