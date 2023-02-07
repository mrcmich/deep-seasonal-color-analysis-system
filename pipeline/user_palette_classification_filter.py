# --- Needed to import modules from other packages
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# ---

from .abstract_filter import AbstractFilter
import torch
import cv2
import torch
from metrics_and_losses import metrics
import utils.utils as utils
import numpy as np
import matplotlib.pyplot as plt
from palette_classification import color_processing, palette
from utils import utils, segmentation_labels
import glob
from typing import OrderedDict

class UserPaletteClassificationFilter(AbstractFilter):
    """
    .. description:: 
    Filter taking as input a tuple (image, segmentation_masks) of pytorch tensors (in the format returned by 
    a segmentation filter) of the user and assigning the corresponding palette object according 
    to color harmony theory.
    """
    
    def __init__(self, reference_palettes):
        """
        .. description:: 
        Class constructor.

        .. inputs::
        reference_palettes: list of palette objects (instances of palette_classification.palette.PaletteRGB) 
        to use as reference for classification.
        """

        self.labels = OrderedDict({ 
            label: segmentation_labels.labels[label] for label in ['skin', 'hair', 'lips', 'eyes'] })

        self.skin_idx = utils.from_key_to_index(self.labels, 'skin')
        self.hair_idx = utils.from_key_to_index(self.labels, 'hair')
        self.lips_idx = utils.from_key_to_index(self.labels, 'lips')
        self.eyes_idx = utils.from_key_to_index(self.labels, 'eyes')
        self.reference_palettes = reference_palettes
        
    def input_type(self):
        """
        .. description::
        Type of couple (image, segmentation masks) the filter expects to receive when executed. The couple
        should be a tuple of two pytorch tensors.
        """
        
        return tuple

    def output_type(self):
        """
        .. description::
        Type of user palette the filter returns when executed.
        """

        return palette.PaletteRGB

    def execute(self, input):
        """
        .. description::
        Method to execute the filter on the provided input. The filter takes the input and returns
        the corresponding palette object.

        .. inputs::
        input: Input of the filter, expected to be the same type returned by method input_type.
        """

        img, masks = input
        masks = color_processing.compute_segmentation_masks(img, self.labels)
        img_masked = color_processing.apply_masks(img, masks)
        
        dominants = color_processing.compute_dominants(
            img_masked, n_candidates=(3, 3, 3, 3), distance_fn=metrics.rmse)
        dominants_palette = palette.PaletteRGB('dominants', dominants)
        
        subtone = palette.compute_subtone(dominants[self.lips_idx])
        intensity = palette.compute_intensity(dominants[self.skin_idx])
        value = palette.compute_value(
            dominants[self.skin_idx], dominants[self.hair_idx], dominants[self.eyes_idx])
        contrast = palette.compute_contrast(dominants[self.hair_idx], dominants[self.eyes_idx])
        
        dominants_palette.compute_metrics_vector(subtone, intensity, value, contrast)
        user_palette = palette.classify_palette(dominants_palette, self.reference_palettes)

        return user_palette