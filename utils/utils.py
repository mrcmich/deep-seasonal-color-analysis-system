import torch
import argparse

# Computes weighted average of specified pytorch tensor. Both tensors should have shape (n, ).
# Returns a pytorch tensor of shape (1,) containing the weighted average as result.
def tensor_weighted_average(tensor, weights):
    assert(type(tensor) == torch.Tensor and type(weights) == torch.Tensor and tensor.shape == weights.shape)

    return (tensor * weights).sum() / weights.sum() 

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

def parse_arguments_train_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', type=str, required=True, choices=["True", "False"], help='If True, validation is used.', metavar='')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs in the validation case', metavar='')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate', metavar='')
    args = parser.parse_args()
    args.evaluate = args.evaluate == "True"
    return args

def parse_arguments_test_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the model weights to use for testing', metavar='')
    return parser.parse_args()
