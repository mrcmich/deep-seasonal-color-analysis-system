import numpy as np
import torch
from typing import OrderedDict


labels = OrderedDict({
    'background': [0, 0, 0],
    'lips': [255, 0, 0],
    'eyes': [0, 255, 0],
    'nose': [0, 0, 255],
    'skin': [128, 128, 128],
    'hair': [255, 255, 0],
    'eyebrows': [255, 0, 255],
    'ears': [0, 255, 255],
    'teeth': [255, 255, 255],
    'beard': [255, 192, 192],
    'sunglasses': [0, 128, 128],
})


# Converts image from (H, W, D) to (D, H, W) by swapping its axes.
def from_HWD_to_DHW(img_HWD):
    return img_HWD.swapaxes(0, 2).swapaxes(1, 2)


# Converts image from (D, H, W) to (H, W, D) by swapping its axes.
def from_DHW_to_HWD(img_DHW):
    return img_DHW.swapaxes(0, 2).swapaxes(0, 1)


def color_mask(img, color_triplet=[0, 0, 0]):
    assert(img.shape[2] == 3 and len(color_triplet) == 3)

    ch0, ch1, ch2 = color_triplet
    mask = (img[:, :, 0] == ch0) * (img[:, :, 1] == ch1) * (img[:, :, 2] == ch2)
    return mask


# Given a segmented image and a dictionary of labels, each corresponding to a different region of an image, returns a numpy 
# array of shape (H, W, n_labels) containing n_labels segmentation masks, where each one is a boolean numpy array 
# of shape (H, W) identifying pixels which belong to the corresponding label.
# ---
# labels: dictionary of labels { label_name (string): color_triplet (list) }.
def compute_segmentation_masks(img_segmented, labels):
    n_labels = len(labels)
    H, W, _ = img_segmented.shape
    masks = np.zeros((H, W, n_labels), dtype=np.float32)

    for idx, label in enumerate(labels):
        label_color = labels[label]
        masks[:, :, idx] = color_mask(img_segmented, label_color)

    return masks
