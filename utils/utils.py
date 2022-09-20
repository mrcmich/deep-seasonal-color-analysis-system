import torch

# Converts image (torch.Tensor instance) from (H, W, D) to (D, H, W) by swapping its axes.
def from_HWD_to_DHW(img_HWD):
    return img_HWD.swapaxes(0, 2).swapaxes(1, 2)

# Converts image (torch.Tensor instance) from (D, H, W) to (H, W, D) by swapping its axes.
def from_DHW_to_HWD(img_DHW):
    return img_DHW.swapaxes(0, 2).swapaxes(0, 1)
    
# Returns the index of a certain key in a dictionary.
def from_key_to_index(dictionary, key):
    return list(dictionary.keys()).index(key)

# Counts the number of learnable parameters of model (parameters whose attribute requires_grad set to True).
def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)