import cv2
import torch
import torchvision.transforms.functional as TF
import random
from utils import utils


class BilateralFilter:
    """
    .. description::
    Custom transform taking a torch.tensor as input and applying a bilateral filter (function cv2.bilateralFilter)
    to it. Returns the result as a torch.tensor.
    """

    def __init__(self, sigma_color, sigma_space, diameter=9):
        self.d = diameter
        self.sigmac = sigma_color
        self.sigmas = sigma_space

    def __call__(self, img: torch.Tensor):
        img_filtered = cv2.bilateralFilter(utils.from_DHW_to_HWD(img).numpy(), self.d, self.sigmac, self.sigmas)
        return utils.from_HWD_to_DHW(torch.from_numpy(img_filtered))


class PartiallyDeterministicHorizontalFlip:
    """
    .. description::
    Custom transform which randomly flips an image (horizontally) with probability p.
    Differently from RandomHorizontalFlip, the same transform is applied for max_seed_count consecutive calls, meaning
    that, for every sequence of max_seed_count calls, the first call is random, whereas the remaining
    max_seed_count - 1 are deterministic.
    This allows for the same RandomHorizontalFlip transform to be applied both to image and target in a dataset,
    when max_seed_count is set to default.

    .. inputs::
    p:      flip probability for the first call of a sequence.
    """
    def __init__(self, p=0.5, max_seed_count=2):
        self.p = p
        self.max_count = max_seed_count
        self.counter = self.max_count

    def __call__(self, img: torch.Tensor):
        if self.counter == self.max_count:
            random.seed()
            self.counter = 0

        random_state = random.getstate()
        n = random.randint(1, 100) / 100
        random.setstate(random_state)
        self.counter += 1

        if n > 1 - self.p:
            return TF.hflip(img)

        return img


class PartiallyDeterministicCenterCrop:
    """
    .. description::
    Custom transform which randomly crops an image with probability p, following the same principle of
    PartiallyDeterministicHorizontalFlip.
    If an image with spatial size (height, width) is cropped, the new spatial dimensions new_height, new_width
    are randomly chosen such that:
    - height_range[0] * height <= new_height <= height_range[1] * height and
    - width_range[0] * width <= new_width <= width_range[1] * width

    .. inputs::
    p:                  crop probability for the first call of a sequence.
    height_range:       tuple containing the minimum and maximum ratios between cropped height and
                        original height (both ratios must be between 0 and 1).
    width_range:        tuple containing the minimum and maximum ratios between cropped width and
                        original width (both ratios must be between 0 and 1).
    """
    def __init__(self, p=0.5, height_range=(0.8, 1), width_range=(0.8, 1), max_seed_count=2):
        self.p = p
        self.max_count = max_seed_count
        self.counter = self.max_count
        self.height_range = height_range
        self.width_range = width_range

    def __call__(self, img: torch.Tensor):
        _, height, width = img.shape
        min_height = int(self.height_range[0] * height)
        max_height = int(self.height_range[1] * height)
        min_width = int(self.width_range[0] * width)
        max_width = int(self.width_range[1] * width)

        if self.counter == self.max_count:
            random.seed()
            self.counter = 0

        random_state = random.getstate()
        n = random.randint(1, 100) / 100
        new_height = random.randint(min_height, max_height)
        new_width = random.randint(min_width, max_width)
        random.setstate(random_state)
        self.counter += 1

        if n > 1 - self.p:
            return TF.center_crop(img, [new_height, new_width])

        return img
