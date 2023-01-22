import cv2
import torch
import torchvision.transforms.functional as TF
import random

# Custom transform taking a torch.tensor as input and applying a bilateral filter 
# (function cv2.bilateralFilter) to it.
# Returns the result as a torch.tensor.
class BilateralFilter:
    def __init__(self, sigma_color, sigma_space, diameter=9):
        self.d = diameter
        self.sigmac = sigma_color
        self.sigmas = sigma_space

    def __call__(self, img: torch.Tensor):
        img_filtered = cv2.bilateralFilter(img.numpy(), self.d, self.sigmac, self.sigmas)
        return torch.from_numpy(img_filtered)

# Custom tranform which randomly flips an image (horizontally).
# Differently from RandomHorizontalFlip, the same transform is applied for max_seed_count consecutive calls, meaning
# that, for every sequence of max_seed_count calls, the first call is random, whereas the remaining 
# max_seed_count - 1 are deterministic.
# This allows for the same RandomHorizontalFlip transform to be applied both to image and target in a dataset, 
# when max_seed_count is set to default.
class PartiallyDeterministicHorizontalFlip:
    def __init__(self, max_seed_count=2):
        self.max_count = max_seed_count
        self.counter = self.max_count

    def __call__(self, img: torch.Tensor):
        if self.counter == self.max_count:
            random.seed()
            self.counter = 0

        random_state = random.getstate()
        n = random.randint(0, 10) / 10
        random.setstate(random_state)
        self.counter += 1

        if n > 0.5:
            return TF.hflip(img)
        
        return img

# to be implemented
class PartiallyDeterministicCenterCrop:
    pass
