#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:1
#SBATCH --error=models/outputs/Deeplabv3_eval.err
#SBATCH --output=models/outputs/Deeplabv3_eval.out
#SBATCH --signal=USR1@600

srun python models_train_pipeline_Deeplabv3.py --evaluate=True
