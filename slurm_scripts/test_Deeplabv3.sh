#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:1
#SBATCH --error=models/outputs/Deeplabv3_test.err
#SBATCH --output=models/outputs/Deeplabv3_test.out

srun python test_pipeline_Deeplabv3.py --weights_path="models/weights/deeplabv3_ccncsa_X_epochs.pth"
