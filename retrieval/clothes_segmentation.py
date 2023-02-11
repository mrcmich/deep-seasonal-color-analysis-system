import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def segment_img_cloth(img_path, save_fig_path=None):
    """
    .. description::
    Function that segments a cloth image from Dress Code Dataset with classic CV operators: adaptive thresholding,
    Canny edge detector and contours tracing.

    .. inputs::
    img_path:                   filepath of the image to segment.
    save_fig_path:              path to folder where to save the images of the segmentation's steps, default is None.

    .. outputs::
    Returns the boolean segmentation mask for the input image.
    if save_fig_path is not None saves in the given directory the images of the segmentation's steps.
    """

    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
    edges = cv2.Canny(img_th, 100, 255)
    edges_dilated = cv2.dilate(edges, None)

    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour = None
    max_len = 0
    for c in contours:
        if c.shape[0] > max_len:
            contour = c
            max_len = c.shape[0]

    contour_img = img_th.copy() * 0
    cv2.drawContours(contour_img, [contour], 0, 255, 3)
    
    seg_mask = contour_img.copy()

    mask = contour_img.copy()
    mask = np.pad(mask, (1, 1))
    cv2.floodFill(seg_mask, mask, seedPoint=(0, 0), newVal=255)
    mask = contour_img.copy()
    mask = np.pad(mask, (1, 1))
    cv2.floodFill(seg_mask, mask, seedPoint=(0, contour_img.shape[1] - 1), newVal=255)

    seg_mask = np.where(seg_mask == 255, 0, 255)
    if save_fig_path is not None:
        plt.subplot(231)
        plt.imshow(img, cmap='gray')
        plt.title('Original')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(232)
        plt.imshow(img_th, cmap='gray')
        plt.title('Thresholded')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(233)
        plt.imshow(edges, cmap='gray')
        plt.title('Edges')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(234)
        plt.imshow(edges_dilated, cmap='gray')
        plt.title('Dilated Edges')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(235)
        plt.imshow(contour_img, cmap='gray')
        plt.title('Contour')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(236)
        plt.imshow(seg_mask, cmap='gray')
        plt.title('Segmented')
        plt.xticks([])
        plt.yticks([])
        
        img_name = "segmented_" + img_path.split("/")[-1]

        plt.savefig(save_fig_path + img_name)
    
    segmentation_mask = seg_mask == 255
    return torch.from_numpy(segmentation_mask).unsqueeze(0)
