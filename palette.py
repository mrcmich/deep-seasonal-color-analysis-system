import numpy as np
import matplotlib.pyplot as plt
import cv2

# Returns a compact string representation of an array by constructing the sequence of its elements.
# e.g. The array [1,1,0,1] results in string '1101'.
def compact_string_(array):
    string = ''

    for element in array:
        string += (str(element))
    
    return string

# Computes subtone (S) by comparing lips color with colors peach and purple according to the following rule:
# if lips_color is closest to peach_color then subtone is 'warm'
# else if lips_color is closest to purple_color then subtone is 'cold'
# ---
# lips_color: numpy array of shape (1, 1, 3).
def compute_subtone(lips_color): 
    peach_color = np.array([255, 230, 182], dtype=np.float32).reshape((1, 1, 3)) / 255
    purple_color = np.array([145, 0, 255], dtype=np.float32).reshape((1, 1, 3)) / 255

    if color_distance(lips_color, peach_color) < color_distance(lips_color, purple_color):
        return 'warm'
    
    return 'cold'

# Computes contrast (C), defined as the brightness difference between hair and eyes.
# ---
# hair_color, eyes_color: numpy arrays of shape (1, 1, 3).
def compute_contrast(hair_color, eyes_color): 
    hair_color_GRAYSCALE = cv2.cvtColor(hair_color, cv2.COLOR_RGB2GRAY).item()
    eyes_color_GRAYSCALE = cv2.cvtColor(eyes_color, cv2.COLOR_RGB2GRAY).item()
    return abs(hair_color_GRAYSCALE - eyes_color_GRAYSCALE) / 255

# Computes intensity (I), defined as skin color saturation.
# ---
# skin_color: numpy array of shape (1, 1, 3).
def compute_intensity(skin_color): 
    skin_color_HSV = cv2.cvtColor((skin_color / 255).astype(np.float32), cv2.COLOR_RGB2HSV)
    return skin_color_HSV[0, 0, 1]

# Computes value (V), defined as the overall brightness of skin, hair and eyes.
# ---
# skin_color, hair_color, eyes_color: numpy arrays of shape (1, 1, 3).
def compute_value(skin_color, hair_color, eyes_color): 
    skin_color_GRAYSCALE = cv2.cvtColor((skin_color / 255).astype(np.float32), cv2.COLOR_RGB2GRAY).item()
    hair_color_GRAYSCALE = cv2.cvtColor((hair_color / 255).astype(np.float32), cv2.COLOR_RGB2GRAY).item()
    eyes_color_GRAYSCALE = cv2.cvtColor((eyes_color / 255).astype(np.float32), cv2.COLOR_RGB2GRAY).item()
    return (skin_color_GRAYSCALE + hair_color_GRAYSCALE + eyes_color_GRAYSCALE) / 3
    
# Converts two RGB colors, represented by numpy arrays of shape (1, 1, 3), in CIELab and then computes
# the euclidean distance between them.
def color_distance(color1_RGB, color2_RGB):
    assert(color1_RGB.shape == (1, 1, 3) and color2_RGB.shape == (1, 1, 3))
    
    color1_CIELab = cv2.cvtColor(color1_RGB, cv2.COLOR_RGB2Lab)
    color2_CIELab = cv2.cvtColor(color2_RGB, cv2.COLOR_RGB2Lab)
    return np.linalg.norm(color1_RGB - color2_RGB)

# Assigns to palette a class taken from reference_palettes, by minimizing the Hamming distance
# between metrics vectors.
def classify_palette(palette, reference_palettes):
    assert(palette.has_metrics_vector())

    min_hamming_distance = -1
    season = PaletteRGB()
    metrics_vector = palette.metrics_vector()
    
    for reference_palette in reference_palettes:
        assert(reference_palette.has_metrics_vector())

        reference_metrics_vector = reference_palette.metrics_vector()

        if reference_metrics_vector[0] != metrics_vector[0]:
            continue

        hamming_distance = (metrics_vector != reference_metrics_vector).sum()

        if min_hamming_distance == -1 or hamming_distance < min_hamming_distance:
            min_hamming_distance = hamming_distance
            season = reference_palette
    
    return season

class PaletteRGB():
    def __init__(self, description='palette', colors=np.zeros((1, 1, 3))):
        assert(type(colors) == np.ndarray)

        self.description_ = str(description)
        self.colors_ = colors.reshape((1, -1, 3))
        self.metrics_vector_ = None
    
    def description(self):
        return self.description_

    def colors(self):
        return self.colors_

    def n_colors(self):
        return self.colors_.shape[1]

    def has_metrics_vector(self):
        return self.metrics_vector_ is not None

    # Returns None if the palette has no metrics vector.
    def metrics_vector(self):
        if self.has_metrics_vector():
            return self.metrics_vector_

        return None

    # Returns a binary numpy array obtained by binarizing a specific combination of metrics values. Each metric value 
    # (except for the subtone) is converted to 1 if above the corresponding threshold or 0 if at or below said threshold.
    # ---
    # thresholds: tuple of thresholds given by (contrast_thresh, intensity_thresh, value_thresh).
    def compute_metrics_vector(self, subtone, intensity, value, contrast, thresholds=(0.5, 0.5, 0.5)):
        sequence = np.zeros((4), dtype=np.uint8)
        contrast_thresh, intensity_thresh, value_thresh = thresholds

        sequence[0] = 'warm' == subtone
        sequence[1] = intensity > intensity_thresh
        sequence[2] = value > value_thresh
        sequence[3] = contrast > contrast_thresh

        self.metrics_vector_ = sequence

    # filepath: directory in which to save the palette.
    def save(self, filepath='', delimiter=';'):
        header = ''
        filename = filepath + self.description_ + '.csv'

        if self.has_metrics_vector():
            header = header + '# metrics vector (SIVC)\n' + compact_string_(self.metrics_vector_) + '\n'

        header = header + '# color data\n'
        np.savetxt(filename, self.colors_.reshape((-1, 3)), header=header, fmt='%u', delimiter=delimiter)
    
    # header: if True, an header with format:
    # >> # metrics vector (SIVC)
    # >> XXXX
    # >> color data
    # , with XXXX being the compact string representation of the palette's metrics vector, is present at the beginning of the file.
    def load(self, filename, header=False, delimiter=';'):
        self.description_ = (str(filename).split('/')[-1]).split('.')[0]

        if header is True:
            file = open(filename)
            file.readline()
            header_data = list(file.readline())
            del header_data[4:]
            self.metrics_vector_ = np.array(header_data, dtype=np.uint8)
            file.close()

        self.colors_ = np.loadtxt(fname=filename, dtype=np.uint8, skiprows=int(header) * 3, delimiter=delimiter).reshape((1, -1, 3))
        return self
    
    # tile_size: size of each color tile (in inches).
    def plot(self, tile_size=5):
        plt.figure(figsize=(tile_size, tile_size * self.n_colors()))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.colors_)
        plt.show()
