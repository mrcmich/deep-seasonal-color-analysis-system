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
    Filter applying semantic segmentation to the input image and returning segmentation masks for said
    input as output. The segmentation model used for predictions can be configured through the model parameter
    of the class constructor ('local' for the less accurate but lighter model, 'cloud' for the more accurate
    but heavier one).
    """
    
    def __init__(self, model):
        assert(model in ['local', 'cloud'])

        n_classes = len(segmentation_labels.labels)
        weights_path = config.WEIGHTS_PATH
        
        if model == 'local':
            model_name = 'fastscnn_ccncsa_best'
            self.model = fast_scnn.FastSCNN(n_classes)
            model_cfg_best = slurm_config.configurations['best']['fastscnn']
        elif model == 'cloud':
            model_name = 'unet_ccncsa_best'
            self.model = unet.UNet(out_channels=n_classes)
            model_cfg_best = slurm_config.configurations['best']['unet']
  
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(weights_path + model_name + '.pth'))
        self.pil_to_tensor = T.Compose([T.PILToTensor()])
        self.transforms = model_cfg_best['image_transform_inference']
       
    def input_type(self):
        """
        .. description::
        Type of input image the filter expects to receive when executed.
        """
        
        return PIL.Image.Image

    def output_type(self):
        """
        .. description::
        Type of segmentation masks the filter returns when executed.
        """

        return torch.Tensor

    def execute(self, input):
        """
        .. description::
        Method to execute the filter on the provided input. The filter takes the input and predicts
        the corresponding segmentation masks using the selected model.

        .. inputs::
        input: Input of the filter, expected to be the same type returned by method input_type.
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input = self.pil_to_tensor(input) / 255
        input = self.transforms(input)
        input = torch.unsqueeze(input, axis=0)
        input = input.to(device)

        with torch.no_grad():
            self.model.eval()
            output = self.model(input)[0]

        channels_max, _ = torch.max(output, dim=1)
        prediction = (output == channels_max.unsqueeze(axis=1))[0]
        
        return prediction
