# --- Needed to import modules from other packages
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# ---

import cv2
import torch
import utils.utils as utils

# Converts two images (torch.Tensor instances) of shape (D, H, W) in CIELab and then computes the RMSE between them.
def rmse(img1, img2):
    assert(img1.shape == img2.shape)

    _, H, W = img1.shape
    img1_np_HWD = utils.from_DHW_to_HWD(img1).numpy()
    img2_np_HWD = utils.from_DHW_to_HWD(img2).numpy()
    img1_CIELab = cv2.cvtColor(img1_np_HWD, cv2.COLOR_RGB2Lab)
    img2_CIELab = cv2.cvtColor(img2_np_HWD, cv2.COLOR_RGB2Lab)

    return (((img1_CIELab - img2_CIELab) ** 2).sum() / (H * W)) ** 0.5

# Returns a pytorch tensor containing the average mIoU along a batch of images.
# ---
# predictions, targets: pytorch tensors of shape (batch_size, n_labels, H, W).
def batch_mIoU(predictions, targets):
    return batch_IoU(predictions, targets).mean()

# Returns a pytorch tensor of shape (n_labels,) containing the IoU for each
# label along a batch of images.
# ---
# predictions, targets: pytorch tensors of shape (batch_size, n_labels, H, W).
def batch_IoU(predictions, targets):
    intersection_cardinality = torch.logical_and(predictions, targets).sum(axis=(2, 3)) 
    union_cardinality = torch.logical_or(predictions, targets).sum(axis=(2, 3)) 
    IoU = intersection_cardinality / union_cardinality
    
    # if there aren't pixels of a certain class in an image, and the model correctly predicts so, 
    # than the IoU should be 1 for that class
    IoU[union_cardinality == 0] = 1.0

    return IoU.mean(axis=0)