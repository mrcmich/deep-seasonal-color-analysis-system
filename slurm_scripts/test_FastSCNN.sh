#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:1
#SBATCH --error=models/outputs/FastSCNN_test.err
#SBATCH --output=models/outputs/FastSCNN_test.out

srun python test_pipeline_FastSCNN.py --weights_path="models/weights/fast_scnn_ccncsa.pth"
