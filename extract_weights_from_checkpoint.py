import os
import torch
from torch import nn
from models.local.FastSCNN.models import fast_scnn
from models.cloud.UNet import unet
from utils import segmentation_labels
from models import config

weights_path = config.WEIGHTS_PATH
fast_scnn_checkpoint_dir = config.CHECKPOINTS_PATH + 'FastSCNN'
unet_checkpoint_dir = config.CHECKPOINTS_PATH + 'UNet'

fast_scnn_name = 'fast_scnn_ccncsa_best'
unet_name = 'unet_ccncsa_best'
n_classes = len(segmentation_labels.labels)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Extract and save FastSCNN's weights
fast_scnn_model = fast_scnn.FastSCNN(n_classes)
fast_scnn_model = nn.DataParallel(fast_scnn_model)
model_state, _, _ = torch.load(os.path.join(fast_scnn_checkpoint_dir, "checkpoint.pt"))
fast_scnn_model.load_state_dict(model_state)
torch.save(fast_scnn_model.state_dict(), weights_path + fast_scnn_name + ".pth")

# Extract and save UNet's weights
unet_model = unet.UNet(out_channels=n_classes)
unet_model = nn.DataParallel(unet_model)
model_state, _, _ = torch.load(os.path.join(unet_checkpoint_dir, "checkpoint.pt"))
unet_model.load_state_dict(model_state)
torch.save(unet_model.state_dict(), weights_path + unet_name + ".pth")
