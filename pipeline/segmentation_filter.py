# --- Needed to import modules from other packages
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# ---

from .abstract_filter import AbstractFilter
import PIL
import torch
from torch import nn
import torchvision.transforms as T
from models import config
from slurm_scripts import slurm_config
from models.local.FastSCNN.models import fast_scnn
from models.cloud.UNet import unet
from utils import segmentation_labels

class SegmentationFilter(AbstractFilter):
    """
    .. description:: 
    Filter applying semantic segmentation to the input image, expected to be the image of a user.
    The filter returns a tuple containing both the input image (converted into a pytorch tensor) and 
    its segmentation masks. The segmentation model used for predictions can be configured through the 
    model parameter of the class constructor ('local' for the less accurate but lighter model, 
    'cloud' for the more accurate but heavier one). Moreover, the filter supports execution both 
    on cpu and gpu. The filter doesn't support the printing of additional information through verbose
    parameter of method execute.
    """
    
    def __init__(self, model):
        assert(model in ['local', 'cloud'])

        n_classes = len(segmentation_labels.labels)
        weights_path = config.WEIGHTS_PATH
        
        if model == 'local':
            model_name = 'fastscnn_ccncsa_best'
            self.model = fast_scnn.FastSCNN(n_classes)
            self.model = nn.DataParallel(self.model)
            model_cfg_best = slurm_config.configurations['best']['fastscnn']
        elif model == 'cloud':
            model_name = 'unet_ccncsa_best'
            self.model = unet.UNet(out_channels=n_classes)
            model_cfg_best = slurm_config.configurations['best']['unet']
  
        self.model.load_state_dict(torch.load(weights_path + model_name + '.pth'))
        self.pil_to_tensor = T.Compose([T.PILToTensor()])
        self.transforms = model_cfg_best['image_transform_inference']
       
    def input_type(self):
        return PIL.Image.Image

    def output_type(self):
        return tuple

    def execute(self, input, device=None, verbose=False):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        input = self.pil_to_tensor(input)
        _, H, W = input.shape
        resize = T.Compose([T.Resize((H, W))])
        input_transformed = self.transforms(input / 255)
        input_transformed = torch.unsqueeze(input_transformed, axis=0)
        input = input.to(device)
        input_transformed = input_transformed.to(device)

        with torch.no_grad():
            self.model = self.model.to(device)
            self.model.eval()
            output = self.model(input_transformed)[0]

        channels_max, _ = torch.max(output, dim=1)
        prediction = (output == channels_max.unsqueeze(axis=1))[0]
        prediction = resize(prediction)

        return (input.to('cpu'), prediction.to('cpu'))
