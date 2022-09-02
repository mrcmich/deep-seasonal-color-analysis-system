# --- Needed to import modules from other packages
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# ---
                
import numpy as np
import torch
import utils.utils as utils
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Converts two RGB colors, represented by pytorch tensors of shape (3, 1, 1), in CIELab and then computes
# the euclidean distance between them.
def color_distance(color1_RGB, color2_RGB):
    assert(color1_RGB.shape == (3, 1, 1) and color2_RGB.shape == (3, 1, 1))
    
    color1_RGB_np_HWD = utils.from_DHW_to_HWD(color1_RGB).numpy()
    color2_RGB_np_HWD = utils.from_DHW_to_HWD(color2_RGB).numpy()
    color1_CIELab = cv2.cvtColor(color1_RGB_np_HWD, cv2.COLOR_RGB2Lab)
    color2_CIELab = cv2.cvtColor(color2_RGB_np_HWD, cv2.COLOR_RGB2Lab)
    return np.linalg.norm(color1_CIELab - color2_CIELab)
    
# Returns a boolean pytorch tensor of shape (H, W), where each pixel (x, y) is True if img[x, y, :] is equal to color_triplet.
# ---
# img: pytorch tensor of shape (3, H, W).
# color_triplet: python list representing a color.
def color_mask(img, color_triplet=[0, 0, 0]):
    assert(img.shape[0] == 3 and len(color_triplet) == 3)

    ch0, ch1, ch2 = color_triplet
    mask = (img[0] == ch0) * (img[1] == ch1) * (img[2] == ch2)
    return mask

# Given a segmented image (torch.Tensor instance) of shape (3, H, W) and a dictionary of labels, each corresponding to a 
# different region of an image, returns a pytorch tensor of shape (n_labels, H, W) containing n_labels segmentation masks, 
# where each one is a boolean pytorch tensor of shape (H, W) identifying pixels which belong to the corresponding label.
# ---
# labels: dictionary of labels { label_name (string): color_triplet (list) }.
def compute_segmentation_masks(img_segmented, labels):
    n_labels = len(labels)
    _, H, W = img_segmented.shape
    masks = torch.zeros((n_labels, H, W), dtype=torch.bool)

    for idx, label in enumerate(labels):
        label_color = labels[label]
        masks[idx, :, :] = color_mask(img_segmented, label_color)

    return masks

# Given an image of shape (3, H, W) and a set of masks represented by a boolean pytorch tensor of shape (n_masks, H, W), applies 
# all masks to the image, resulting in a new image with shape (n_masks, 3, H, W).
# ---
def apply_masks(img, masks):
    assert(img.shape[1] == masks.shape[1] and img.shape[2] == masks.shape[2])

    img_masked = img * masks.unsqueeze(axis=1)
    return img_masked.to(torch.uint8)

# Given a masked image of shape (n_masks, 3, H, W) and a distance function computing a distance measure between two images, 
# returns a pytorch tensor of shape (n_masks, 3, 1, 1) containing the dominant colors associated to each mask. When comparing candidates,
# brighter colors are favored for skin, hair, lips dominants and darker colors are favored for the eyes dominant (this is done by appropriately)
# weighting the provided distance measure).
# ---
# n_candidates: tuple of length n_masks specifying how many candidates to consider for each mask when looking for a dominant.
def compute_dominants(img_masked, n_candidates, distance_fn, debug=False):
    assert(img_masked.shape[0] >= 4 and img_masked.shape[0] == len(n_candidates))

    IMG_MASKED_EYES_IDX = 3
    n_masks, _, H, W = img_masked.shape
    dominants = []

    for i in range(n_masks):
        img_masked_i = img_masked[i]
        max_brightness_i = cv2.cvtColor(utils.from_DHW_to_HWD(img_masked_i).numpy(), cv2.COLOR_RGB2GRAY).max()
        kmeans = KMeans(n_clusters=n_candidates[i], random_state=99)
        mask_i = np.logical_not(color_mask(img_masked_i))                
        img_masked_i_flattened = utils.from_DHW_to_HWD(img_masked_i).reshape((H * W, -1)) / 255
        img_masked_i_flattened_sample = shuffle(img_masked_i_flattened, random_state=99, n_samples=round(0.6 * H * W))
        kmeans.fit(img_masked_i_flattened_sample)
        candidates = torch.round(torch.from_numpy(kmeans.cluster_centers_) * 255).to(torch.uint8)
        reconstructions = mask_i * candidates.unsqueeze(axis=2).unsqueeze(axis=3)

        min_reconstruction_error = -1 
        dominant = torch.zeros((3, 1, 1), dtype=torch.uint8)

        for j, reconstruction_j in enumerate(reconstructions):
            if candidates[j].sum() < 20 or candidates[j].sum() > 600:
                continue
            
            average_brightness_j = cv2.cvtColor(utils.from_DHW_to_HWD(reconstruction_j).numpy(), cv2.COLOR_RGB2GRAY).mean()
            reconstruction_error_j = distance_fn(img_masked_i, reconstruction_j).item()

            if i == IMG_MASKED_EYES_IDX:
                # decrease RMSE of darker colors when computing eyes dominant
                reconstruction_error_j *= (average_brightness_j / max_brightness_i)
            else:
                # decrease RMSE of brighter colors when computing other dominants
                reconstruction_error_j /= (average_brightness_j / max_brightness_i)

            # debug
            if debug is True:
                r, g, b = candidates[j]
                print(f'Candidate: ({r},{g},{b}), Weighted Reconstruction Error: {reconstruction_error_j}')
                plt.figure(figsize=(20, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(utils.from_DHW_to_HWD(reconstruction_j))
                plt.subplot(1, 2, 2)
                plt.imshow(utils.from_DHW_to_HWD(img_masked_i))
                plt.show() 

            if min_reconstruction_error == -1 or reconstruction_error_j < min_reconstruction_error:
                min_reconstruction_error = reconstruction_error_j
                dominant = candidates[j]
            
        dominants.append(dominant.tolist())
    
    return torch.tensor(dominants, dtype=torch.uint8).reshape((n_masks, 3, 1, 1))