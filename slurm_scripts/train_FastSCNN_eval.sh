#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:1
#SBATCH --error=models/outputs/FastSCNN_eval.err
#SBATCH --output=models/outputs/FastSCNN_eval.out
#SBATCH --signal=USR1@600

srun python train_pipeline_FastSCNN.py --evaluate=True
