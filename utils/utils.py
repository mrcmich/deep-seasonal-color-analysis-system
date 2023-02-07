import torch
import argparse
import matplotlib.pyplot as plt
from palette_classification import color_processing
from utils import segmentation_labels, model_names
from slurm_scripts import slurm_config
from models import config


def tensor_weighted_average(tensor, weights):
    """
    .. description::
    Computes weighted average of specified pytorch tensor. Both tensors should have shape (n, ).

    .. outputs::
    Returns a pytorch tensor of shape (1,) containing the weighted average as result.
    """
    assert (type(tensor) == torch.Tensor and type(weights) == torch.Tensor and tensor.shape == weights.shape)

    return (tensor * weights).sum() / weights.sum()


def from_HWD_to_DHW(img_HWD):
    """
    .. description::
    Converts image (torch.Tensor instance) from (H, W, D) to (D, H, W) by swapping its axes.
    """
    return img_HWD.swapaxes(0, 2).swapaxes(1, 2)


def from_DHW_to_HWD(img_DHW):
    """
    .. description::
    Converts image (torch.Tensor instance) from (D, H, W) to (H, W, D) by swapping its axes.
    """
    return img_DHW.swapaxes(0, 2).swapaxes(0, 1)


def from_key_to_index(dictionary, key):
    """
    .. description::
    Returns the index of a certain key in a dictionary.
    """
    return list(dictionary.keys()).index(key)


def count_learnable_parameters(model):
    """
    .. description::
    Counts the number of learnable parameters of model (parameters whose attribute requires_grad set to True).
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_training_or_hpo_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, choices=list(slurm_config.configurations.keys()),
                        help='Which training configuration to use', metavar='')
    parser.add_argument('--model_name', type=str, required=True, choices=list(model_names.MODEL_NAMES.keys()),
                        help='Which model to use', metavar='')
    parser.add_argument('--evaluate', type=str, required=True, choices=["True", "False"],
                        help='If True, validation is used.', metavar='')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of epochs in the validation case', metavar='')
    args = parser.parse_args()
    args.evaluate = args.evaluate == "True"

    # verify that the configuration of the model exists for the specified slurm configuration
    if args.model_name not in slurm_config.configurations[args.config]:
        parser.error(f'Model {args.model_name} not available for configuration {args.config}.')

    return args


def parse_retrieval_arguments(train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_model", default='ViT-B-32', choices=list(model_names.CLIP_MODELS_PRETRAINED.keys()), type=str)
    parser.add_argument("--category", default='all', choices=["all", "dresses", "upper_body", "lower_body"], type=str)
    parser.add_argument("--dataroot", type=str, default=config.DRESSCODE_PATH_ON_LAB_SERVER)
    parser.add_argument("--phase", default="test", choices=["train", "test"], type=str)
    parser.add_argument("--order", default="unpaired", choices=["paired", "unpaired"], type=str)
    parser.add_argument("--query", default="dress", choices=["dress", "upper_body", "lower_body"], type=str)
    parser.add_argument("--k", default=5, type=int)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)

    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--shuffle", default=True, action='store_true', help='shuffle input data')
    args = parser.parse_args()
    return args


def parse_save_weights_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, choices=list(slurm_config.configurations.keys()),
                        help='Which training configuration to use', metavar='')
    parser.add_argument('--model_name', type=str, required=True, choices=list(model_names.MODEL_NAMES.keys()),
                        help='Which model to use', metavar='')
    parser.add_argument('--parallel', type=str, required=True, choices=["True", "False"],
                        help='If True, the model has been trained with DataParallel.', metavar='')
    args = parser.parse_args()
    args.parallel = args.parallel == "True"

    # verify that the configuration of the model exists for the specified slurm configuration
    if args.model_name not in slurm_config.configurations[args.config]:
        parser.error(f'Model {args.model_name} not available for configuration {args.config}.')

    return args


def plot_random_examples(device, model, dataset, n_examples=1, figsize=(12, 6)):
    """
    .. description::
    Given a model and a dataset, plots predictions (and corresponding targets) for n_examples random images.
    """
    _, tH, tW = dataset[0][0].shape
    n_classes = len(segmentation_labels.labels)
    random_images = torch.zeros((n_examples, 3, tH, tW))
    random_targets = torch.zeros((n_examples, n_classes, tH, tW))

    for i in range(n_examples):
        random_idx = torch.randint(high=len(dataset), size=(1,))
        random_image, random_target = dataset[random_idx]
        random_images[i] = random_image
        random_targets[i] = random_target

    with torch.no_grad():
        model.eval()
        random_images = random_images.to(device)
        random_output = model(random_images)[0]

    channels_max, _ = torch.max(random_output, dim=1)
    random_predictions = (random_output == channels_max.unsqueeze(axis=1)).to('cpu')

    for i in range(n_examples):
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.title('Ground Truth')
        plt.imshow(from_DHW_to_HWD(
            color_processing.colorize_segmentation_masks(random_targets[i], segmentation_labels.labels)))
        plt.subplot(1, 2, 2)
        plt.title('Prediction')
        plt.imshow(from_DHW_to_HWD(
            color_processing.colorize_segmentation_masks(random_predictions[i], segmentation_labels.labels)))
