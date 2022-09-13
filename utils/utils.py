import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Converts image (torch.Tensor instance) from (H, W, D) to (D, H, W) by swapping its axes.
def from_HWD_to_DHW(img_HWD):
    return img_HWD.swapaxes(0, 2).swapaxes(1, 2)

# Converts image (torch.Tensor instance) from (D, H, W) to (H, W, D) by swapping its axes.
def from_DHW_to_HWD(img_DHW):
    return img_DHW.swapaxes(0, 2).swapaxes(0, 1)

# Given an image (torch.Tensor instance) img of shape (3, H, W) returns the image obtained by applying gamma correction to img.
def gamma_correction(img, gamma):
    img_gamma_corrected = ((img / 255) ** (1 / gamma)) * 255
    return torch.round(img_gamma_corrected).to(torch.uint8)

# Loads a RGB image from disk given its filename. When gamma_decode is set to True, the image is gamma decoded with a gamma value
# computed from its metadata (if not available, default_gamma_decoding is used instead as a gamma value). Returns
# a tuple (image, gamma), where image is a pytorch tensor of shape (3, H, W) representing the loaded RGB image 
# and gamma is the gamma value used during gamma decoding (always 1.0 when gamma_decode is set to False).
def load_image(img_filename, gamma_decode=False, default_gamma_decoding=2.2):
    img = Image.open(img_filename).convert('RGB')

    if gamma_decode is True:
        gamma_decoding = (1 / img.info['gamma']) if 'gamma' in img.info else default_gamma_decoding
    else:
        gamma_decoding = 1.0

    img = from_HWD_to_DHW(torch.from_numpy(np.array(img)))
    img_decoded = gamma_correction(img, gamma_decoding)

    return img_decoded, gamma_decoding

# Given a RGB image (torch.Tensor instance) img of shape (3, H, W),
# shows a new image obtained by applying gamma encoding to img.
def show_image(img, gamma_encoding=1/2.2):
    img_encoded = gamma_correction(img, gamma_encoding)
    plt.imshow(from_DHW_to_HWD(img_encoded))

# Returns the index of a certain key in a dictionary.
def from_key_to_index(dictionary, key):
    return list(dictionary.keys()).index(key)

# Counts the number of learnable parameters of model (parameters whose attribute requires_grad set to True).
def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)