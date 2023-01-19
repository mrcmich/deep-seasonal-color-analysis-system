#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:1
#SBATCH --error=models/outputs/Deeplabv3_complete.err
#SBATCH --output=models/outputs/Deeplabv3_complete.out
#SBATCH --signal=USR1@600

srun python train_pipeline_Deeplabv3.py --evaluate=False
