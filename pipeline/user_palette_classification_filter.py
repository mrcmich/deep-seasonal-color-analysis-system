# --- Needed to import modules from other packages
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# ---

from .abstract_filter import AbstractFilter
from metrics_and_losses import metrics
import utils.utils as utils
from palette_classification import color_processing, palette
from utils import utils, segmentation_labels

class UserPaletteClassificationFilter(AbstractFilter):
    """
    .. description:: 
    Filter taking as input a tuple (image, segmentation_masks) of pytorch tensors (in the format returned by 
    a segmentation filter) of the user and assigning the corresponding palette object according 
    to color harmony theory. The filter returns said palette object as output. Please note that the
    filter doesn't support execution on gpu, and thus the device parameter of method execute has no
    effect on execution. The filter supports the printing of additional information through verbose
    parameter of method execute.
    """
    
    def __init__(self, reference_palettes, thresholds=(0.200, 0.422, 0.390)):
        """
        .. inputs::
        reference_palettes: list of palette objects (instances of palette_classification.palette.PaletteRGB) 
                            to use as reference for classification.
        thresholds:         tuple of thresholds to use when binarizing values of metrics contrast, 
                            intensity, value (values must be between 0 and 1).
        """

        assert(0 <= thresholds[0] <= 1 and 0 <= thresholds[1] <= 1 and 0 <= thresholds[2] <= 1)

        relevant_labels = ['skin', 'hair', 'lips', 'eyes']
        self.relevant_indexes = [ 
            utils.from_key_to_index(segmentation_labels.labels, label) for label in relevant_labels ]
        self.skin_idx, self.hair_idx, self.lips_idx, self.eyes_idx = (0, 1, 2, 3)
        self.reference_palettes = reference_palettes
        self.thresholds = thresholds
        
    def input_type(self):
        return tuple

    def output_type(self):
        return palette.PaletteRGB

    def execute(self, input, device=None, verbose=False):
        img, masks = input
        relevant_masks = masks[self.relevant_indexes, :, :]
        img_masked = color_processing.apply_masks(img, relevant_masks)
        
        dominants = color_processing.compute_user_embedding(
            img_masked, n_candidates=(3, 3, 3, 3), distance_fn=metrics.rmse, debug=verbose)
        dominants_palette = palette.PaletteRGB('dominants', dominants)
        
        hair_dominant = dominants[self.hair_idx] if relevant_masks[self.hair_idx].sum() > 0 else None
        subtone = palette.compute_subtone(dominants[self.lips_idx])
        intensity = palette.compute_intensity(dominants[self.skin_idx])
        value = palette.compute_value(
            dominants[self.skin_idx], hair_dominant, dominants[self.eyes_idx])
        contrast = palette.compute_contrast(hair_dominant, dominants[self.eyes_idx])
        
        with_contrast = dominants_palette.compute_metrics_vector(
            subtone, intensity, value, contrast, self.thresholds)
        user_palette = palette.classify_user_palette(
            dominants_palette, self.reference_palettes, with_contrast)
        
        if verbose is True:
            print(dominants_palette.description())
            dominants_palette.plot()

        return user_palette