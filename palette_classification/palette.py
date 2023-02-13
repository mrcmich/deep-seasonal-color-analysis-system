# --- Needed to import modules from other packages
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# ---

import torch
import numpy as np
import matplotlib.pyplot as plt
from . import color_processing
import utils.utils as utils
import cv2


def compact_string_(array):
    """
    .. description::
    Returns a compact string representation of an array by constructing the sequence of its elements.
    e.g. The array [1,1,0,1] results in string '1101'.
    """
    string = ''

    for element in array:
        string += (str(element))
    
    return string


def compute_subtone(lips_color):
    """
    .. description::
    Computes subtone (S) by comparing lips color with colors peach and purple according to the following rule:
    if lips_color is closest to peach_color then subtone is 'warm'
    else if lips_color is closest to purple_color then subtone is 'cold'

    .. inputs::
    lips_color:     pytorch tensor of shape (3, 1, 1).
    """
    assert(lips_color.shape == (3, 1, 1))

    peach_color = torch.tensor([255, 230, 182], dtype=torch.uint8).reshape(lips_color.shape)
    purple_color = torch.tensor([210, 120, 180], dtype=torch.uint8).reshape(lips_color.shape)

    if color_processing.color_distance(lips_color, peach_color) < color_processing.color_distance(lips_color,
                                                                                                  purple_color):
        return 'warm'
    
    return 'cold'


def compute_contrast(hair_color, eyes_color):
    """
    .. description::
    Computes contrast (C), defined as the brightness difference between hair and eyes. Returns None if
    hair_color is None (meaning that it's unavailable).

    .. inputs::
    hair_color, eyes_color: pytorch tensors of shape (3, 1, 1).
    """

    if hair_color is None:
        return None

    hair_color_np_HWD = utils.from_DHW_to_HWD(hair_color).numpy()
    eyes_color_np_HWD = utils.from_DHW_to_HWD(eyes_color).numpy()
    hair_color_HSV = cv2.cvtColor((hair_color_np_HWD / 255).astype(np.float32), cv2.COLOR_RGB2HSV)
    eyes_color_HSV = cv2.cvtColor((eyes_color_np_HWD / 255).astype(np.float32), cv2.COLOR_RGB2HSV)

    return abs(hair_color_HSV[0, 0, 2] - eyes_color_HSV[0, 0, 2])


def compute_intensity(skin_color):
    """
    .. description::
    Computes intensity (I), defined as skin color saturation.

    .. inputs::
    skin_color: pytorch tensor of shape (3, 1, 1).
    """
    skin_color_np_HWD = utils.from_DHW_to_HWD(skin_color).numpy()
    skin_color_HSV = cv2.cvtColor((skin_color_np_HWD / 255).astype(np.float32), cv2.COLOR_RGB2HSV)
    return skin_color_HSV[0, 0, 1]


def compute_value(skin_color, hair_color, eyes_color):
    """
    .. description::
    Computes value (V), defined as the overall brightness of skin, hair and eyes. If hair_color is None,
    then the value is computed using only skin_color and eyes_color.

    .. inputs::
    skin_color, hair_color, eyes_color: pytorch tensors of shape (3, 1, 1).
    """

    skin_color_np_HWD = utils.from_DHW_to_HWD(skin_color).numpy()
    eyes_color_np_HWD = utils.from_DHW_to_HWD(eyes_color).numpy()
    skin_color_HSV = cv2.cvtColor((skin_color_np_HWD / 255).astype(np.float32), cv2.COLOR_RGB2HSV)
    eyes_color_HSV = cv2.cvtColor((eyes_color_np_HWD / 255).astype(np.float32), cv2.COLOR_RGB2HSV)

    if hair_color is None:
        return (skin_color_HSV[0, 0, 2] + eyes_color_HSV[0, 0, 2]) / 2

    hair_color_np_HWD = utils.from_DHW_to_HWD(hair_color).numpy()
    hair_color_HSV = cv2.cvtColor((hair_color_np_HWD / 255).astype(np.float32), cv2.COLOR_RGB2HSV)

    return (skin_color_HSV[0, 0, 2] + hair_color_HSV[0, 0, 2] + eyes_color_HSV[0, 0, 2]) / 3


def classify_user_palette(user_palette, reference_palettes, with_contrast=True):
    """
    .. description::
    Assigns to user_palette a class taken from reference_palettes, by minimizing the Hamming distance
    between metrics vectors. If user_palette has metrics vector with form [s0, s1, s2, None] (signaling that
    it wasn't possible to compute the contrast metrics), then the distance is computed considering only 
    the first three elements of the two metrics vectors.

    .. inputs::
    with_contrast:  boolean indicating wether to use the contrast metric during the classification process.
    """
    assert(user_palette.has_metrics_vector())

    min_hamming_distance = -1
    season = PaletteRGB()
    metrics_vector = user_palette.metrics_vector()

    if with_contrast is False:
        metrics_vector = metrics_vector[:-1]
    
    for reference_palette in reference_palettes:
        assert(reference_palette.has_metrics_vector())

        reference_metrics_vector = reference_palette.metrics_vector() 
        
        if with_contrast is False:
            reference_metrics_vector = reference_metrics_vector[:-1]

        if reference_metrics_vector[0] != metrics_vector[0]:
            continue

        hamming_distance = (metrics_vector != reference_metrics_vector).sum()

        if min_hamming_distance == -1 or hamming_distance < min_hamming_distance:
            min_hamming_distance = hamming_distance
            season = reference_palette
    
    return season

def classify_cloth_palette(cloth_palette, reference_palettes, distance_type='avg'):
    assert distance_type in ['min', 'max', 'avg']

    closest_palette = None
    min_palette_distance = -1

    for reference_palette in reference_palettes:
        palette_distance = cloth_palette.distance_from(reference_palette, distance_type)

        if min_palette_distance == -1 or palette_distance < min_palette_distance:
            min_palette_distance = palette_distance
            closest_palette = reference_palette
    
    return closest_palette

class PaletteRGB:
    """
    .. description::
    Class representing a palette of RGB colors, each described by a pytorch tensor of shape (3, 1, 1).

    .. inputs::
    colors: pytorch tensor of shape (n_colors, 3, 1, 1).
    """

    def __init__(self, description='palette', colors=torch.zeros((1, 3, 1, 1))):
        assert(type(colors) == torch.Tensor)

        self.description_ = str(description)
        self.colors_ = colors.swapaxes(0, 1).reshape((3, 1, -1))
        self.metrics_vector_ = None
    
    def description(self):
        return self.description_

    def colors(self):
        """
        .. description::
        Returns palette colors, represented by a pytorch tensor of shape (3, 1, n_colors).
        """
        return self.colors_

    def n_colors(self):
        return self.colors_.shape[2]

    def has_metrics_vector(self):
        return self.metrics_vector_ is not None

    def metrics_vector(self):
        """
        .. description::
        Returns None if the palette has no metrics vector.
        """
        if self.has_metrics_vector():
            return self.metrics_vector_

        return None

    def compute_metrics_vector(self, subtone, intensity, value, contrast, thresholds=(0.5, 0.5, 0.5)):
        """
        .. description::
        Computes the metrics vector of the palette, represented by a binary pytorch tensor obtained by 
        binarizing a specific combination of metrics values. Each metric value (except for the subtone) 
        is converted to 1 if above the corresponding threshold or 0 if at or below said
        threshold. Returns False if contrast is None, to indicate that the last element of the metrics vector
        must be ignored, True otherwise.

        .. inputs::
        thresholds:     tuple of thresholds given by (contrast_thresh, intensity_thresh, value_thresh).
        """
        sequence = np.zeros(4, dtype=np.uint8)
        contrast_thresh, intensity_thresh, value_thresh = thresholds

        sequence[0] = 'warm' == subtone
        sequence[1] = intensity > intensity_thresh
        sequence[2] = value > value_thresh
        sequence[3] = 0 if contrast is None else contrast > contrast_thresh

        self.metrics_vector_ = torch.from_numpy(sequence)

        if contrast is None:
            return False
        
        return True

    def save(self, filepath='', delimiter=';'):
        """
        .. inputs::
        filepath:   directory in which to save the palette.
        """
        header = ''
        filename = filepath + self.description_ + '.csv'

        if self.has_metrics_vector():
            header = header + '# metrics vector (SIVC)\n' + compact_string_(self.metrics_vector_) + '\n'

        header = header + '# color data\n'
        np.savetxt(filename, utils.from_DHW_to_HWD(self.colors_).reshape((-1, 3)).numpy(),
                   header=header, fmt='%u', delimiter=delimiter)

    def load(self, filename, header=False, delimiter=';'):
        """
        .. description::
        header: if True, an header with format:
        >> # metrics vector (SIVC)
        >> XXXX
        >> color data
        with XXXX being the compact string representation of the palette's metrics vector, is present at the beginning
        of the file.
        """
        self.description_ = (str(filename).split('/')[-1]).split('.')[0]

        if header is True:
            file = open(filename)
            file.readline()
            header_data = list(file.readline())
            del header_data[4:]
            self.metrics_vector_ = torch.from_numpy(np.array(header_data, dtype=np.uint8))
            file.close()

        self.colors_ = torch.from_numpy(np.loadtxt(fname=filename, dtype=np.uint8, skiprows=int(header) * 3,
                                                   delimiter=delimiter).reshape((1, -1, 3)))
        self.colors_ = utils.from_HWD_to_DHW(self.colors_)
        return self

    def plot(self, tile_size=5):
        """
        .. inputs::
        tile_size:  size of each color tile (in inches).
        """
        plt.figure(figsize=(tile_size, tile_size * self.n_colors()))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(utils.from_DHW_to_HWD(self.colors_).numpy())
        plt.show()

    def distance_from(self, other_palette, type='avg'):
        """
        .. description::
        Method computing a distance measure between calling palette and other_palette. Because the
        two palettes could have a different number of colors, the method first matches each color
        of the calling palette with one color in other_palette (the one having closest hue), and then
        computes a distance between colors for each match. From the resulting vector of distances, a 
        single distance value is returned by applying a function between min(), max(), average().

        .. inputs::
        type:   type of function to apply in order to get a single distance value; 
                must be in ['min', 'max', 'avg'].
        """
        assert type in ['min', 'max', 'avg']

        distances = []

        for color_idx in range(self.n_colors()):
            min_hue_distance = -1
            closest_color = None
            color = torch.unsqueeze(self.colors_[:, :, color_idx], dim=1)
            color_np_HWD = utils.from_DHW_to_HWD(color).numpy()
            color_hue = cv2.cvtColor((color_np_HWD / 255).astype(np.float32), cv2.COLOR_RGB2HSV)[0, 0, 0]

            for other_color_idx in range(other_palette.n_colors()):
                other_color = torch.unsqueeze(other_palette.colors()[:, :, other_color_idx], dim=1)
                other_color_np_HWD = utils.from_DHW_to_HWD(other_color).numpy()
                other_color_hue = cv2.cvtColor(
                    (other_color_np_HWD / 255).astype(np.float32), cv2.COLOR_RGB2HSV)[0, 0, 0]
                hue_distance = 180 - abs(abs(color_hue - other_color_hue) - 180)

                if min_hue_distance == -1 or hue_distance < min_hue_distance:
                    min_hue_distance = hue_distance
                    closest_color = other_color
            
            min_color_distance = color_processing.color_distance(color, closest_color)
            distances.append(min_color_distance)
        
        distances = torch.tensor(distances)

        if type == 'min':
            distance = distances.min()
        elif type == 'max':
            distance = distances.max()
        elif type == 'avg':
            distance = distances.mean()

        return distance.item()



