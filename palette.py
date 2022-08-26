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

# Returns a binary numpy array obtained by binarizing a specific combination of metrics values. Each metric value 
# (except for the subtone) is converted to 1 if above the corresponding threshold or 0 if at or below said threshold.
# ---
# thresholds: tuple of thresholds given by (contrast_thresh, intensity_thresh, value_thresh).
def compute_metrics_vector(subtone, intensity, value, contrast, thresholds=(0.5, 0.5, 0.5)):
    sequence = np.zeros((4), dtype=np.uint8)
    contrast_thresh, intensity_thresh, value_thresh = thresholds

    sequence[0] = 'warm' == subtone
    sequence[1] = intensity > intensity_thresh
    sequence[2] = value > value_thresh
    sequence[3] = contrast > contrast_thresh

    return sequence

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

    def set_metrics_vector(self, metrics_vector):
        assert(metrics_vector.shape == (4,) and ((0 <= metrics_vector) * (metrics_vector <= 1)).all())
        self.metrics_vector_ = metrics_vector.astype(np.uint8)

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
