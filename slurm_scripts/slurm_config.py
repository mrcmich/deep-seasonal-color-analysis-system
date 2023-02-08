import torchvision.transforms as T
from models import config
from ray import tune
import os
import torch
from utils import custom_transforms, model_names


class SlurmConfig:
    def __init__(self,
                 input_size, image_transform, 
                 image_transform_inference, # image transform specific for inference
                 target_transform,
                 use_weighted_loss,  # True in order to use weighted cross entropy loss
                 optimizer,
                 local_dir,  # parameter local_dir of function tune.run
                 tunerun_cfg,  # dictionary to be passed to parameter config of function tune.run
                 is_hpo_cfg  # specifies whether the configuration is for hpo or not)
                ):
        self.config_dict_ = {
            'input_size': input_size,
            'image_transform': image_transform,
            'image_transform_inference': image_transform_inference,
            'target_transform': target_transform,
            'weighted_loss': use_weighted_loss,
            'optimizer': optimizer,
            'local_dir': local_dir,
            'tunerun_cfg': tunerun_cfg,
            'hpo_cfg': is_hpo_cfg,
        }

    def config_dict(self):
        return self.config_dict_


# === Global configurations ===

# config for training of demo models
GLOBAL_INPUT_SIZE_TRAINING_DEMO = (256, 256)
GLOBAL_IMAGE_TRANSFORM_TRAINING_DEMO = T.Compose([
    T.Resize(GLOBAL_INPUT_SIZE_TRAINING_DEMO),
    T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)
])
GLOBAL_CFG_TRAINING_DEMO = SlurmConfig(
    GLOBAL_INPUT_SIZE_TRAINING_DEMO,
    GLOBAL_IMAGE_TRANSFORM_TRAINING_DEMO,
    GLOBAL_IMAGE_TRANSFORM_TRAINING_DEMO,
    T.Compose([T.Resize(GLOBAL_INPUT_SIZE_TRAINING_DEMO)]),
    False,
    torch.optim.Adam,
    config.DEMO_PATH,
    {
        "lr": 0.01,
        "lr_scheduler": "none",
        "batch_size": 32,
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.DEMO_PATH) + '/'
    },
    False
).config_dict()

# === FastSCNN-specific configurations ===

# config for hpo
FASTSCNN_INPUT_SIZE_HPO = (256, 256)
FASTSCNN_CFG_HPO = SlurmConfig(
    FASTSCNN_INPUT_SIZE_HPO,
    T.Compose([
        T.Resize(FASTSCNN_INPUT_SIZE_HPO),
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    T.Compose([
        T.Resize(FASTSCNN_INPUT_SIZE_HPO),
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    T.Compose([T.Resize(FASTSCNN_INPUT_SIZE_HPO)]),
    True,
    torch.optim.Adam,
    config.HPO_PATH,
    {
        "lr": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
        "lr_scheduler": tune.grid_search(["none", "linear"]),
        "batch_size": tune.grid_search([16, 32]),
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.HPO_PATH) + '/'
    },
    True
).config_dict()

# config for training with best hyperparameter values from hpo
FASTSCNN_INPUT_SIZE_TRAINING_BEST = (256, 256)
FASTSCNN_CFG_TRAINING_BEST = SlurmConfig(
    FASTSCNN_INPUT_SIZE_TRAINING_BEST,
    T.Compose([
        T.Resize(FASTSCNN_INPUT_SIZE_TRAINING_BEST),
        custom_transforms.BilateralFilter(sigma_color=50, sigma_space=100, diameter=7),
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    T.Compose([
        T.Resize(FASTSCNN_INPUT_SIZE_TRAINING_BEST),
        custom_transforms.BilateralFilter(sigma_color=50, sigma_space=100, diameter=7),
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    T.Compose([T.Resize(FASTSCNN_INPUT_SIZE_TRAINING_BEST)]),
    True,
    torch.optim.Adam,
    config.CHECKPOINTS_PATH,
    {
        "lr": 0.001,
        'lr_scheduler': "none",
        "batch_size": 16,
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.CHECKPOINTS_PATH) + '/'
    },
    False
).config_dict()

# === UNet-specific configurations ===

# config for hpo
UNET_INPUT_SIZE_HPO = (256, 256)
T.Compose([
    T.Resize(UNET_INPUT_SIZE_HPO),
    T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)])
UNET_CFG_HPO = SlurmConfig(
    UNET_INPUT_SIZE_HPO,
    T.Compose([
        T.Resize(UNET_INPUT_SIZE_HPO),
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    T.Compose([
        T.Resize(UNET_INPUT_SIZE_HPO),
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    T.Compose([T.Resize(UNET_INPUT_SIZE_HPO)]),
    True,
    torch.optim.Adam,
    config.HPO_PATH,
    {
        "lr": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
        "lr_scheduler": tune.grid_search(["none", "linear"]),
        "batch_size": tune.grid_search([16, 32]),
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.HPO_PATH) + '/'
    },
    True
).config_dict()

# config for training with best hyperparameter values from hpo
UNET_INPUT_SIZE_TRAINING_BEST = (256, 256)
UNET_CFG_TRAINING_BEST = SlurmConfig(
    UNET_INPUT_SIZE_TRAINING_BEST,
    T.Compose([
        T.ColorJitter(brightness=0.25, contrast=0.25),
        T.Resize(UNET_INPUT_SIZE_TRAINING_BEST),
        custom_transforms.BilateralFilter(sigma_color=50, sigma_space=100, diameter=7),
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    T.Compose([
        T.Resize(UNET_INPUT_SIZE_TRAINING_BEST),
        custom_transforms.BilateralFilter(sigma_color=50, sigma_space=100, diameter=7),
        T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)]),
    T.Compose([T.Resize(UNET_INPUT_SIZE_TRAINING_BEST)]),
    True,
    torch.optim.Adam,
    config.CHECKPOINTS_PATH,
    {
        "lr": 1e-4,
        'lr_scheduler': "none",
        "batch_size": 16,
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.CHECKPOINTS_PATH) + '/'
    },
    False
).config_dict()

# slurm configurations
MODEL_NAMES_LIST = list(model_names.MODEL_NAMES.keys())
SLURM_CFG_HPO = {'fastscnn': FASTSCNN_CFG_HPO, 'unet': UNET_CFG_HPO}
SLURM_CFG_TRAINING_BEST = {'fastscnn': FASTSCNN_CFG_TRAINING_BEST, 'unet': UNET_CFG_TRAINING_BEST}
SLURM_CFG_TRAINING_DEMO = {model_name: GLOBAL_CFG_TRAINING_DEMO for model_name in MODEL_NAMES_LIST}

# dictionary containing all slurm configurations
configurations = {'demo': SLURM_CFG_TRAINING_DEMO, 'hpo': SLURM_CFG_HPO, 'best': SLURM_CFG_TRAINING_BEST}
