import torchvision.transforms as T
from models import config
from ray import tune
import os
import torch

# === FastSCNN configurations ===

# cfg 1
FASTSCNN_INPUT_SIZE_01 = (256, 256)
FASTSCNN_HPO_CFG_01 = {
    "lr": tune.grid_search([1e-4, 1e-2]),
    "batch_size": tune.grid_search([16, 32, 64]),
    "start_factor": tune.grid_search([0.3, 0.5]),
    "from_checkpoint": False,
    "checkpoint_dir": os.path.abspath("./" + config.HPO_PATH + "FastSCNN")
}
FASTSCNN_CFG_01 = {
    'image_transform': T.Compose([
        T.Resize(FASTSCNN_INPUT_SIZE_01), 
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    'target_transform': T.Compose([T.Resize(FASTSCNN_INPUT_SIZE_01)]),
    'hpo_cfg': FASTSCNN_HPO_CFG_01,
    'optimizer': torch.optim.Adam,
    'lr_scheduler': torch.optim.lr_scheduler.LinearLR,
    'local_dir': config.HPO_PATH + 'FastSCNN'
}

# === UNet configurations ===

# cfg 1
UNET_INPUT_SIZE_01 = (256, 256)
UNET_HPO_CFG_01 = {
    "lr": tune.grid_search([1e-4, 1e-2]),
    "batch_size": tune.grid_search([16, 32, 64]),
    "start_factor": tune.grid_search([0.3, 0.5]),
    "from_checkpoint": False,
    "checkpoint_dir": os.path.abspath("./" + config.HPO_PATH + "UNet")
}
UNET_CFG_01 = {
    'image_transform': T.Compose([
        T.Resize(UNET_INPUT_SIZE_01), 
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    'target_transform': T.Compose([T.Resize(UNET_INPUT_SIZE_01)]),
    'hpo_cfg': UNET_HPO_CFG_01,
    'optimizer': torch.optim.AdamW,
    'lr_scheduler': torch.optim.lr_scheduler.LinearLR,
    'local_dir': config.HPO_PATH + 'UNet'
}

# slurm configuration dictionary
SLURM_CFG_01 = { 'fastscnn': FASTSCNN_CFG_01, 'unet': UNET_CFG_01 }