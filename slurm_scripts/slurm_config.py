import torchvision.transforms as T
from models import config
from ray import tune
import os
import torch
from utils import custom_transforms

# === FastSCNN configurations ===

# config for hpo
FASTSCNN_INPUT_SIZE_HPO = (256, 256)
FASTSCNN_CFG_HPO = {
    'image_transform': T.Compose([
        T.Resize(FASTSCNN_INPUT_SIZE_HPO), 
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    'target_transform': T.Compose([T.Resize(FASTSCNN_INPUT_SIZE_HPO)]),
    'optimizer': torch.optim.Adam,
    'local_dir': config.HPO_PATH + 'FastSCNN',
    'tunerun_cfg': { # dictionary to be passed to parameter config of function tune.run
        "lr": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
        "lr_scheduler": tune.grid_search(["none", "linear"]),
        "batch_size": tune.grid_search([16, 32]),
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.HPO_PATH + "FastSCNN")
    },
}

# config for training with best hyperparameter values from hpo
FASTSCNN_CENTER_CROP_TRAINING_BEST = custom_transforms.PartiallyDeterministicCenterCrop(p=0.5)
FASTSCNN_INPUT_SIZE_TRAINING_BEST = (256, 256)
FASTSCNN_CFG_TRAINING_BEST = {
    'n_epochs': 20,
    'input_size': FASTSCNN_INPUT_SIZE_TRAINING_BEST,
    'image_transform': T.Compose([
        FASTSCNN_CENTER_CROP_TRAINING_BEST,
        T.ColorJitter(brightness=0.25, contrast=0.25), 
        T.Resize(FASTSCNN_INPUT_SIZE_TRAINING_BEST), 
        custom_transforms.BilateralFilter(sigma_color=50, sigma_space=100, diameter=7),
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    'target_transform': T.Compose([
        FASTSCNN_CENTER_CROP_TRAINING_BEST,
        T.Resize(FASTSCNN_INPUT_SIZE_TRAINING_BEST)]),
    'optimizer': torch.optim.Adam,
    'local_dir': config.CHECKPOINTS_PATH + 'FastSCNN',
    'tunerun_cfg': {
        "lr": ...,
        'lr_scheduler':  "linear",
        "batch_size": ...,
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.CHECKPOINTS_PATH + "FastSCNN")
    }
}


# === UNet configurations ===

# config for hpo
UNET_INPUT_SIZE_HPO = (256, 256)
UNET_CFG_HPO = {
    'image_transform': T.Compose([
        T.Resize(UNET_INPUT_SIZE_HPO), 
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    'target_transform': T.Compose([T.Resize(UNET_INPUT_SIZE_HPO)]),
    'optimizer': torch.optim.Adam,
    'local_dir': config.HPO_PATH + 'UNet',
    'tunerun_cfg': {
        "lr": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
        "lr_scheduler": tune.grid_search(["none", "linear"]),
        "batch_size": tune.grid_search([16, 32]),
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.HPO_PATH + "UNet")
    }
}

# config for training with best hyperparameter values from hpo
UNET_INPUT_SIZE_TRAINING_BEST = (256, 256)
UNET_CFG_TRAINING_BEST = {
    'n_epochs': 30,
    'input_size': UNET_INPUT_SIZE_TRAINING_BEST,
    'image_transform': T.Compose([
        T.Resize(UNET_INPUT_SIZE_TRAINING_BEST), 
        custom_transforms.BilateralFilter(sigma_color=50, sigma_space=100, diameter=7),
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    'target_transform': T.Compose([T.Resize(UNET_INPUT_SIZE_TRAINING_BEST)]),
    'optimizer': torch.optim.Adam,
    'local_dir': config.CHECKPOINTS_PATH + 'UNet',
    'tunerun_cfg': {
        "lr": ...,
        'lr_scheduler':  "linear",
        "batch_size": ...,
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.CHECKPOINTS_PATH + "UNet")
    }
}

# slurm configuration dictionaries
# n.b. define new dictionaries for additional configurations.
SLURM_CFG_HPO = { 'fastscnn': FASTSCNN_CFG_HPO, 'unet': UNET_CFG_HPO }
SLURM_CFG_TRAINING_BEST = { 'fastscnn': FASTSCNN_CFG_TRAINING_BEST, 'unet': UNET_CFG_TRAINING_BEST }
