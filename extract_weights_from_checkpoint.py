import os
import torch
from torch import nn
from models.local.FastSCNN.models import fast_scnn
from models.local.CGNet.model import CGNet
from models.local.LEDNet.models import lednet
from models.cloud.UNet import unet
from models.cloud.Deeplabv3 import deeplabv3
from utils import segmentation_labels, utils, model_names
from models import config
from slurm_scripts import slurm_config

def save_weights(args):
    model_name = args.model_name
    parallel = args.parallel
    weights_path = config.WEIGHTS_PATH
    cfg_name = args.config
    cfg = slurm_config.configurations[cfg_name]
    model_cfg = cfg[model_name]
    model_cfg['local_dir'] = model_cfg['local_dir'] + model_names.MODEL_NAMES[model_name]
    
    if cfg_name == "best":
        model_cfg["local_dir"] += "/complete"

    n_classes = len(segmentation_labels.labels)
    # instantiating model
    if model_name == "fastscnn":
        model = fast_scnn.FastSCNN(n_classes)
    elif model_name == "cgnet":
        model = CGNet.Context_Guided_Network(classes=n_classes)
    elif model_name == "lednet":
        model = lednet.LEDNet(num_classes=n_classes, output_size=slurm_config.GLOBAL_INPUT_SIZE_TRAINING_DEMO)
    elif model_name == "unet":
        model = unet.UNet(out_channels=n_classes)
    elif model_name == "deeplab":
        model = deeplabv3.deeplabv3_resnet50(num_classes=n_classes)
    else:
        raise Exception("model not supported.")
    
    if parallel:
        model = nn.DataParallel(model)

    # Extract and save weights
    model_state, _ = torch.load(os.path.join(model_cfg["local_dir"], "checkpoint.pt"))
    model.load_state_dict(model_state)
    torch.save(model.state_dict(), weights_path + model_name + f"_ccncsa_{args.config}.pth")

if __name__ == "__main__":
    args = utils.parse_save_weights_arguments()
    
    save_weights(args)
