import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import time
import math
from ray.air import session
import os
from utils import utils, segmentation_labels


def training_or_testing_epoch_(device, model, data_loader, score_fn, loss_fn=None, training=False, optimizer=None,
                               class_weights=None):
    """
    .. description::
    A single epoch of training of model on data_loader. If training is set to False, validation/testing is carried out
    instead, leaving the model's parameters unchanged. In this case, optimizer and loss_fn aren't necessary.

    .. inputs::
    device:         device on which to load data ('cpu' for cpu).
    score_fn:       function to evaluate a batch of predictions against the corresponding batch of targets.

    .. outputs::
    Returns a tuple (average_loss, average_score) representing the average values of loss (zero if not defined) and
    score along data_loader's batches.
    """
    if training:
        assert (optimizer is not None and loss_fn is not None)

    cum_loss = 0.0
    cum_scores = torch.tensor(0.0, dtype=torch.float32)

    for batch_inputs, batch_targets in data_loader:
        if training:
            optimizer.zero_grad()

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.float().to(device)
        batch_outputs = model(batch_inputs)[0]
        channels_max, _ = torch.max(batch_outputs, dim=1)
        batch_predictions = (batch_outputs == channels_max.unsqueeze(axis=1)).float().to(device)

        if loss_fn is not None:
            loss = loss_fn(batch_outputs, batch_targets)
            cum_loss += loss.item()

        score = score_fn(batch_predictions, batch_targets, class_weights)
        cum_scores = cum_scores + score

        if training:
            loss.backward()
            optimizer.step()

    average_loss = cum_loss / len(data_loader)
    average_score = cum_scores / len(data_loader)

    return average_loss, average_score


def train_model(config, device, model, dataset, n_epochs, score_fn, loss_fn, optimizer, class_weights=None,
                num_workers=(0, 0), evaluate=False):
    """
    .. description::
    Function for performing either hpo or training of model on dataset. If evaluate is set to True, the dataset is split
    85/15 into a training set and a validation set, otherwise the entire dataset is used.

    .. inputs::
    config:             dict with hyperparameters
    device:             device on which to load data and model ('cpu' for cpu).
    dataset:            instance of class extending torch.utils.data.Dataset.
    score_fn:           function to evaluate a batch of predictions against the corresponding batch of targets.
    num_workers:        integer or tuple representing the number of workers to use to load train and validation data.
    from_checkpoint:    bool that establishes whether to resume a previous run from the last checkpoint saved.
    """
    model_on_device = model.to(device)

    # model parameters
    batch_size = config["batch_size"]
    optimizer = optimizer(model_on_device.parameters(), lr=config["lr"])
    try:
        lr_scheduler = config["lr_scheduler"]
        if lr_scheduler == "linear":
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        elif lr_scheduler == "none":
            lr_scheduler = None
        else:
            raise Exception("lr_scheduler not supported.")
    except KeyError:
        lr_scheduler = None

    # start from a checkpoint
    if config["from_checkpoint"]:
        model_state, optimizer_state, lr_scheduler_state = torch.load(
            os.path.join(config["checkpoint_dir"], "checkpoint.pt"))
        model_on_device.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        lr_scheduler.load_state_dict(lr_scheduler_state)

    # create workers and DataLoaders
    if type(num_workers) is tuple:
        num_workers_train = num_workers[0]
        num_workers_test = num_workers[1]
    else:
        num_workers_train = num_workers_test = num_workers

    if evaluate:
        n_train_samples = round(0.85 * len(dataset))
        n_val_samples = len(dataset) - n_train_samples
        dataset_train, dataset_val = random_split(
            dataset, lengths=[n_train_samples, n_val_samples], generator=torch.Generator().manual_seed(99))
        dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers_train)
        dl_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers_test)
    else:
        dl_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers_train)
        dl_val = None

    for _ in range(n_epochs):
        model_on_device.train()
        average_train_loss, average_train_score = training_or_testing_epoch_(
            device, model_on_device, dl_train, score_fn, loss_fn, training=True, optimizer=optimizer,
            class_weights=class_weights)
        average_train_score = average_train_score.mean().item()

        if dl_val is not None:
            model_on_device.eval()
            with torch.no_grad():
                average_val_loss, average_val_score = training_or_testing_epoch_(device, model_on_device, dl_val,
                                                                                 score_fn, loss_fn,
                                                                                 class_weights=class_weights)
            average_val_score = average_val_score.mean().item()

        # save a checkpoint
        if lr_scheduler is not None:
            torch.save(
                (model_on_device.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict()),
                os.path.join(config["checkpoint_dir"], "checkpoint.pt"))
            lr_scheduler.step()
        else:
            torch.save(
                (model_on_device.state_dict(), optimizer.state_dict()),
                os.path.join(config["checkpoint_dir"], "checkpoint.pt"))

        # report metrics to Ray Tune
        if evaluate:
            session.report({"train_loss": average_train_loss, "train_score": average_train_score,
                            "val_loss": average_val_loss, "val_score": average_val_score})
        else:
            session.report({"train_loss": average_train_loss, "train_score": average_train_score})

    model_on_device.eval()


def test_model(device, model, dataset, batch_size, score_fn, num_workers=0):
    """
    .. description::
    Function for testing model on dataset.

    .. inputs::
    device:         device on which to load data and model ('cpu' for cpu).
    dataset:        instance of class extending torch.utils.data.Dataset.
    score_fn:       function to evaluate a batch of predictions against the corresponding batch of targets.
    num_workers:    number of workers to use when loading test data.

    .. outputs::
    Returns the average score along dataset's batches.
    """
    dl_test = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    model_on_device = model.to(device)
    model_on_device.eval()

    clock_start = time.time()

    with torch.no_grad():
        _, average_score = training_or_testing_epoch_(device, model_on_device, dl_test, score_fn)

    clock_end = time.time()

    print(f'Device: {device}.')
    print(f'\nInference completed in around {math.ceil(clock_end - clock_start)} seconds.')

    return average_score


def plot_training_results(results_dict, plotsize, filepath=None, train_fmt='g', val_fmt='b'):
    """
    .. description::
    Plots training results, as returned from function train_model.

    .. inputs::
    results_dict:           dictionary with keys average_train_loss, average_val_loss, average_train_score,
                            average_val_score, each identifying a python list of losses/scores.
    plotsize:               tuple representing the size (in inches) of the figure containing results_dict's plots.
    train_fmt:              format string for plots of training metrics.
    val_fmt:                format string for plots of validation metrics.

    .. outputs::
    Returns the average score along dataset's batches.
    """

    assert (
            'average_train_loss' in results_dict and
            'average_val_loss' in results_dict and
            'average_train_score' in results_dict and
            'average_val_score' in results_dict
    )

    legend = ['Train', 'Val']
    xlabel = 'Epoch'
    n_epochs = len(results_dict['average_train_loss'])

    plt.figure(figsize=plotsize)

    # average losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epochs + 1), results_dict['average_train_loss'], train_fmt,
             range(1, n_epochs + 1), results_dict['average_val_loss'], val_fmt)
    plt.title('Average Loss')
    plt.xlabel(xlabel)
    plt.ylabel('Loss')
    plt.legend(legend)

    # average scores
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_epochs + 1), results_dict['average_train_score'], train_fmt, range(1, n_epochs + 1),
             results_dict['average_val_score'], val_fmt)
    plt.title('Average Score')
    plt.xlabel(xlabel)
    plt.ylabel('Score')
    plt.legend(legend)

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()


def print_IoU_report(batch_IoU, class_weights=None):
    """
    .. description::
    Given the batch_IoU (as computed by function test_model when score_fn is set to metrics.batch_IoU),
    prints a report of the IoU of each class and the computed mIoU. If class_weights is passed, the function
    also computes the weighted mIoU.

    .. inputs::
    class_weights: optional pytorch tensor of shape (n_classes,) for the computation of the weighted mIoU.
    """
    label_names = list(segmentation_labels.labels.keys())
    batch_IoU_with_labels = {label: score for label, score in list(zip(label_names, batch_IoU.tolist()))}
    batch_mIoU = batch_IoU.mean().item()

    for label in batch_IoU_with_labels:
        print(f'batch_IoU_{label}: {batch_IoU_with_labels[label]}')

    print(f'batch_mIoU={batch_mIoU}')

    if class_weights is not None:
        batch_weighted_mIoU = utils.tensor_weighted_average(batch_IoU, class_weights).item()
        print(f'batch_weighted_mIoU={batch_weighted_mIoU}')
