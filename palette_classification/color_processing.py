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
import warnings
from skimage import color

def color_distance(color1_RGB, color2_RGB):
    """
    .. description::
    Converts two RGB colors, represented by pytorch tensors of shape (3, 1, 1), in CIELab and then computes
    the euclidean distance between them.
    """
    assert(color1_RGB.shape == (3, 1, 1) and color2_RGB.shape == (3, 1, 1))

    color1_RGB_np_HWD = utils.from_DHW_to_HWD(color1_RGB).numpy()
    color2_RGB_np_HWD = utils.from_DHW_to_HWD(color2_RGB).numpy()
    color1_CIELab = color.rgb2lab(color1_RGB_np_HWD)
    color2_CIELab = color.rgb2lab(color2_RGB_np_HWD)
    return np.linalg.norm(color1_CIELab - color2_CIELab)
    
def color_mask(img, color_triplet=[0, 0, 0]):
    """
    .. description::
    Returns a boolean pytorch tensor of shape (H, W), where each pixel (x, y) is True if img[x, y, :] is equal
    to color_triplet.

    .. inputs::
    img:                pytorch tensor of shape (3, H, W).
    color_triplet:      python list representing a color.
    """
    assert(img.shape[0] == 3 and len(color_triplet) == 3)

    ch0, ch1, ch2 = color_triplet
    mask = (img[0] == ch0) * (img[1] == ch1) * (img[2] == ch2)
    return mask


def compute_segmentation_masks(img_segmented, labels):
    """
    .. description::
    Given a segmented image (torch.Tensor instance) of shape (3, H, W) and a dictionary of labels, each corresponding
    to a different region of an image, returns a pytorch tensor of shape (n_labels, H, W) containing n_labels
    segmentation masks, where each one is a boolean pytorch tensor of shape (H, W) identifying pixels which belong to
    the corresponding label.

    .. inputs::
    labels: dictionary of labels { label_name (string): color_triplet (list) }.
    """
    n_labels = len(labels)
    _, H, W = img_segmented.shape
    masks = torch.zeros((n_labels, H, W), dtype=torch.bool)

    for idx, label in enumerate(labels):
        label_color = labels[label]
        masks[idx, :, :] = color_mask(img_segmented, label_color)

    return masks

def erode_segmentation_mask(segmentation_mask, kernel_size):
    """
    .. description::
    Function taking as input a segmentation mask - a boolean pytorch tensor with shape (1, H, W) - and 
    applying the erosion operator to said mask. Returns the mask obtained after the erosion process.

    .. inputs:: 
    kernel_size: size of the erosion kernel.
    """

    assert(segmentation_mask.shape[0] == 1 and len(segmentation_mask.shape) == 3)

    _, H, W = segmentation_mask.shape
    kernel = cv2.getStructuringElement(shape=0, ksize=(kernel_size, kernel_size))

    extended_segmentation_mask = segmentation_mask * torch.ones(3, H, W)
    img_binarized = torch.where(extended_segmentation_mask == True, 255, 0).to(torch.uint8)
    img_binarized_eroded = cv2.erode(utils.from_DHW_to_HWD(img_binarized).numpy(), kernel=kernel)
    img_binarized_eroded = utils.from_HWD_to_DHW(torch.from_numpy(img_binarized_eroded))
    img_binarized_eroded = torch.unsqueeze(img_binarized_eroded, dim=0)
    img_binarized_eroded = img_binarized_eroded.sum(axis=1)
    segmentation_mask_eroded = torch.where(img_binarized_eroded > 0, True, False)

    return segmentation_mask_eroded

def colorize_segmentation_masks(segmentation_masks, labels):
    """
    .. description::
    Given a boolean pytorch tensor of shape (n_labels, H, W) containing n_labels segmentation masks
    and a dictionary of labels, returns a RGB image (as a pytorch tensor of shape (3, H, W)) obtained by
    assigning a color from labels to each mask.

    .. inputs::
    labels: dictionary of labels { label_name (string): color_triplet (list) }.
    """

    assert(segmentation_masks.shape[0] == len(labels))

    n_labels = segmentation_masks.shape[0]
    color_tensor = torch.tensor(list(labels.values()), dtype=torch.uint8).reshape((n_labels, 3))
    img_colorized = segmentation_masks.unsqueeze(axis=1) * color_tensor.unsqueeze(axis=2).unsqueeze(axis=3)
    return img_colorized.sum(axis=0).to(torch.uint8)


def apply_masks(img, masks):
    """
    .. description::
    Given an image of shape (3, H, W) and a set of masks represented by a boolean pytorch tensor of shape
    (n_masks, H, W), applies all masks to the image, resulting in a new image with shape (n_masks, 3, H, W).
    """
    assert(img.shape[1] == masks.shape[1] and img.shape[2] == masks.shape[2])

    img_masked = img * masks.unsqueeze(axis=1)
    return img_masked.to(torch.uint8)

def compute_candidate_dominants_and_reconstructions_(img_masked, n_candidates, return_recs=True):
    """
    Function taking as input a masked image img_masked (pytorch tensor of shape (3, H, W)) and using clustering to
    identify the n_candidates candidate dominant colors in said image. If return_recs is True, the function
    also computes and returns the reconstructions of the masked image obtained by replacing all 
    non-black pixels with one of the n_candidates candidate dominants found.
    Returns a tuple (candidates, None) or (candidates, reconstructions) as output.
    """

    _, H, W = img_masked.shape
    kmeans = KMeans(n_clusters=n_candidates, n_init=10, random_state=99)
    mask_i = np.logical_not(color_mask(img_masked))                
    img_masked_i_flattened = utils.from_DHW_to_HWD(img_masked).reshape((H * W, -1)) / 255
        
    # silencing kmeans convergence warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='Number of distinct clusters*')
        kmeans.fit(img_masked_i_flattened)

    candidates = torch.round(torch.from_numpy(kmeans.cluster_centers_) * 255).to(torch.uint8)
    
    if return_recs is True:
        reconstructions = mask_i * candidates.unsqueeze(axis=2).unsqueeze(axis=3)
        return candidates, reconstructions

    return candidates, None

def compute_user_embedding(img_masked, n_candidates, distance_fn, debug=False, eyes_idx=3):
    """
    .. description::
    Given a masked image of shape (4, 3, H, W) and a distance function computing a distance measure between two
    images, returns a pytorch tensor of shape (4, 3, 1, 1) containing the dominant colors associated to each mask.
    The four dominants are ordered as follows: skin dominant, hair dominant, lips dominant, eyes dominant.
    When comparing candidates, brighter colors are favored for skin, hair, lips dominants and darker colors are favored
    for the eyes dominant (this is done by appropriately) weighting the provided distance measure).

    .. inputs::
    n_candidates:   tuple of length 4 specifying how many candidates to consider for each mask when looking for a
                    dominant.
    eyes_idx:       index of mask selecting the eyes of the user in img_masked.
    """
    assert(img_masked.shape[:2] == (4, 3) and len(n_candidates) == 4)

    _, _, H, W = img_masked.shape
    dominants = []

    for i in range(4):
        max_brightness_i = cv2.cvtColor(utils.from_DHW_to_HWD(
            img_masked[i] / 255).numpy().astype(np.float32), cv2.COLOR_RGB2HSV)[:, :, 2].max()
        candidates, reconstructions = compute_candidate_dominants_and_reconstructions_(
            img_masked[i], n_candidates[i])

        min_reconstruction_error = -1 
        dominant = torch.zeros((3,), dtype=torch.uint8)

        for j, reconstruction_j in enumerate(reconstructions):
            if candidates[j].sum() < 20 or candidates[j].sum() > 700:
                continue
            
            average_brightness_j = cv2.cvtColor(utils.from_DHW_to_HWD(
                reconstruction_j / 255).numpy().astype(np.float32), cv2.COLOR_RGB2HSV)[:, :, 2].mean()
            reconstruction_error_j = distance_fn(img_masked[i], reconstruction_j).item()

            if i == eyes_idx:
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
                plt.imshow(utils.from_DHW_to_HWD(img_masked[i]))
                plt.show() 

            if min_reconstruction_error == -1 or reconstruction_error_j < min_reconstruction_error:
                min_reconstruction_error = reconstruction_error_j
                dominant = candidates[j]
            
        dominants.append(dominant.tolist())
    
    return torch.tensor(dominants, dtype=torch.uint8).reshape((4, 3, 1, 1))

def compute_cloth_embedding(cloth_img_masked, max_length=10, ignored_colors=[]):
    assert(cloth_img_masked.shape[:2] == (1, 3))

    _, _, H, W = cloth_img_masked.shape
    embedding = []
    
    cloth_colors, _ = compute_candidate_dominants_and_reconstructions_(
        cloth_img_masked[0], max_length + 1, return_recs=False)

    for color in cloth_colors:
        for ignored_color in ignored_colors:
            color_triplet = color.tolist()
            
            if color_triplet != ignored_color:
                embedding.append(color_triplet)

    return torch.tensor(embedding, dtype=torch.uint8).reshape(len(embedding), 3, 1, 1)
