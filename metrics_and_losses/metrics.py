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

# Computes the average mIoU over a batch of images.
# ---
# predictions, targets: pytorch tensors of shape (batch_size, n_labels, H, W).
def average_mIoU(predictions, targets):
    intersection_cardinality = torch.logical_and(predictions, targets).sum(axis=(2, 3)) 
    union_cardinality = torch.logical_or(predictions, targets).sum(axis=(2, 3)) 
    IoU = intersection_cardinality / union_cardinality
    return IoU.mean().item()