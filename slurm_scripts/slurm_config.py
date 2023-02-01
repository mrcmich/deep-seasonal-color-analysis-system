import torchvision.transforms as T
from models import config
from ray import tune
import os
import torch
from utils import custom_transforms

# === FastSCNN configurations ===

# config for hpo
FASTSCNN_INPUT_SIZE_HPO = (256, 256)
FASTSCNN_HPO_CFG_HPO = {
    "lr": tune.grid_search([1e-4, 1e-2]),
    "batch_size": tune.grid_search([16, 32, 64]),
    "start_factor": tune.grid_search([0.3, 0.5]),
    "from_checkpoint": False,
    "checkpoint_dir": os.path.abspath("./" + config.HPO_PATH + "FastSCNN")
}
FASTSCNN_CFG_HPO = {
    'image_transform': T.Compose([
        T.Resize(FASTSCNN_INPUT_SIZE_HPO), 
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    'target_transform': T.Compose([T.Resize(FASTSCNN_INPUT_SIZE_HPO)]),
    'hpo_cfg': FASTSCNN_HPO_CFG_HPO,
    'optimizer': torch.optim.Adam,
    'lr_scheduler': torch.optim.lr_scheduler.LinearLR,
    'local_dir': config.HPO_PATH + 'FastSCNN'
}

# config for tuning
FASTSCNN_CENTER_CROP_TUNING = custom_transforms.PartiallyDeterministicCenterCrop(p=0.5)
FASTSCNN_INPUT_SIZE_TUNING = (512, 512)
FASTSCNN_HPO_CFG_TUNING = {
    "lr": 0.01,
    "batch_size": 32,
    "start_factor": 0.3,
    "from_checkpoint": False,
    "checkpoint_dir": os.path.abspath("./" + config.CHECKPOINTS_PATH + "FastSCNN")
}
FASTSCNN_CFG_TUNING = {
    'n_epochs': 20,
    'input_size': FASTSCNN_INPUT_SIZE_TUNING,
    'image_transform': T.Compose([
        FASTSCNN_CENTER_CROP_TUNING,
        T.ColorJitter(brightness=0.25, contrast=0.25), 
        T.Resize(FASTSCNN_INPUT_SIZE_TUNING), 
        custom_transforms.BilateralFilter(sigma_color=50, sigma_space=100, diameter=7),
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    'target_transform': T.Compose([
        FASTSCNN_CENTER_CROP_TUNING,
        T.Resize(FASTSCNN_INPUT_SIZE_TUNING)]),
    'hpo_cfg': FASTSCNN_HPO_CFG_TUNING,
    'optimizer': torch.optim.Adam,
    'lr_scheduler': torch.optim.lr_scheduler.LinearLR,
    'local_dir': config.CHECKPOINTS_PATH + 'FastSCNN'
}

# === UNet configurations ===

# config for hpo
UNET_INPUT_SIZE_HPO = (256, 256)
UNET_HPO_CFG_HPO = {
    "lr": tune.grid_search([1e-4, 1e-2]),
    "batch_size": tune.grid_search([16, 32, 64]),
    "start_factor": tune.grid_search([0.3, 0.5]),
    "from_checkpoint": False,
    "checkpoint_dir": os.path.abspath("./" + config.HPO_PATH + "UNet")
}
UNET_CFG_HPO = {
    'image_transform': T.Compose([
        T.Resize(UNET_INPUT_SIZE_HPO), 
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    'target_transform': T.Compose([T.Resize(UNET_INPUT_SIZE_HPO)]),
    'hpo_cfg': UNET_HPO_CFG_HPO,
    'optimizer': torch.optim.AdamW,
    'lr_scheduler': torch.optim.lr_scheduler.LinearLR,
    'local_dir': config.HPO_PATH + 'UNet'
}

# config for tuning
UNET_INPUT_SIZE_TUNING = (256, 256)
UNET_HPO_CFG_TUNING = {
    "lr": 0.0001,
    "batch_size": 16,
    "start_factor": 0.5,
    "from_checkpoint": False,
    "checkpoint_dir": os.path.abspath("./" + config.CHECKPOINTS_PATH + "UNet")
}
UNET_CFG_TUNING = {
    'n_epochs': 20,
    'input_size': UNET_INPUT_SIZE_TUNING,
    'image_transform': T.Compose([
        T.Resize(UNET_INPUT_SIZE_TUNING), 
        custom_transforms.BilateralFilter(sigma_color=50, sigma_space=100, diameter=7),
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    'target_transform': T.Compose([T.Resize(UNET_INPUT_SIZE_TUNING)]),
    'hpo_cfg': UNET_HPO_CFG_TUNING,
    'optimizer': torch.optim.AdamW,
    'lr_scheduler': torch.optim.lr_scheduler.LinearLR,
    'local_dir': config.CHECKPOINTS_PATH + 'UNet'
}

# slurm configuration dictionaries
SLURM_CFG_HPO = { 'fastscnn': FASTSCNN_CFG_HPO, 'unet': UNET_CFG_HPO }
SLURM_CFG_TUNING = { 'fastscnn': FASTSCNN_CFG_TUNING, 'unet': UNET_CFG_TUNING }