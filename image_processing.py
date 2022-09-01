import numpy as np
import cv2

# Converts image from (H, W, D) to (D, H, W) by swapping its axes.
def from_HWD_to_DHW(img_HWD):
    return img_HWD.swapaxes(0, 2).swapaxes(1, 2)

# Converts image from (D, H, W) to (H, W, D) by swapping its axes.
def from_DHW_to_HWD(img_DHW):
    return img_DHW.swapaxes(0, 2).swapaxes(0, 1)

# Converts two images of shape (H, W, D) in CIELab and then computes the RMSE between them.
def rmse(img1, img2):
    assert(img1.shape == img2.shape)

    H, W, _ = img1.shape
    img1_CIELab = cv2.cvtColor(img1, cv2.COLOR_RGB2Lab)
    img2_CIELab = cv2.cvtColor(img2, cv2.COLOR_RGB2Lab)

    return (((img1_CIELab - img2_CIELab) ** 2).sum() / (H * W)) ** 0.5

# Returns a boolean numpy array of shape (H, W), where each pixel (x, y) is True if img[x, y, :] is equal to color_triplet.
# ---
# img: numpy array with shape (H, W, 3).
# color_triplet: python list representing a color.
def color_mask(img, color_triplet=[0, 0, 0]):
    assert(img.shape[2] == 3 and len(color_triplet) == 3)

    img_DHW = from_HWD_to_DHW(img)
    ch0, ch1, ch2 = color_triplet
    mask = (img_DHW[0] == ch0) * (img_DHW[1] == ch1) * (img_DHW[2] == ch2)
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

# Given an image of shape (H, W, D) and a set of masks represented by a boolean numpy array of shape (H, W, n_masks), applies 
# all masks to the image, resulting in a new image with shape (n_masks, H, W, D).
# ---
def apply_masks(img, masks):
    assert(img.shape[0] == masks.shape[0] and img.shape[1] == masks.shape[1])

    img_masked = np.expand_dims(img, axis=0) * np.expand_dims(from_HWD_to_DHW(masks), axis=3)
    return img_masked.astype(np.uint8)