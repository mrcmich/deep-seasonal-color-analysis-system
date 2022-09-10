import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import time
import math

# A single epoch of training of model on data_loader. If training is set to False, validation/testing is carried out instead, 
# leaving the model's parameters unchanged. In this case, optimizer and loss_fn aren't necessary. Returns a tuple 
# (average_loss, average_score) representing the average values of loss (zero if not defined) and score along data_loader's batches.
# ---
# device: device on which to load data ('cpu' for cpu).
# score_fn: function to be used to evaluate a batch of predictions against the corresponding batch of targets.
def training_or_testing_epoch_(device, model, data_loader, score_fn, loss_fn=None, training=False, optimizer=None):
    if training is True:
        assert(optimizer is not None and loss_fn is not None)

    cum_loss = 0.0
    cum_scores = torch.tensor(0.0, dtype=torch.float32)

    for batch_inputs, batch_targets in data_loader:
        if training is True:
            optimizer.zero_grad()

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.float().to(device)
        batch_outputs = model(batch_inputs)[0]
        channels_max, _ = torch.max(batch_outputs, axis=1)
        batch_predictions = (batch_outputs == channels_max.unsqueeze(axis=1)).float().to(device)
        
        if loss_fn is not None:
            loss = loss_fn(batch_outputs, batch_targets)
            cum_loss += loss.item()

        score = score_fn(batch_predictions, batch_targets)
        cum_scores = cum_scores + score
        
        if training is True:
            loss.backward()
            optimizer.step()
    
    average_loss = cum_loss / len(data_loader)
    average_score = cum_scores / len(data_loader)

    return average_loss, average_score

# Function for training model on dataset. If evaluate is set to True, the dataset is split 85/15 into a training set
# and a validation set, otherwise the entire dataset is used for training. If verbose is set to True, additional info is
# printed to console during training. Returns a dictionary with keys average_train_loss, average_val_loss, average_train_score,
# average_val_score, each identifying a python list which for each epoch stores the average loss/score along dataset's batches.
# ---
# device: device on which to load data and model ('cpu' for cpu).
# dataset: instance of class extending torch.utils.data.Dataset.
# score_fn: function to be used to evaluate a batch of predictions against the corresponding batch of targets.
def train_model(
    device, model, dataset, batch_size, n_epochs, score_fn, loss_fn, optimizer, lr_scheduler=None, num_workers=(0, 0), evaluate=False, verbose=False):
    
    model_on_device = model.to(device)
    
    num_workers_train = 0
    num_workers_test = 0
    if type(num_workers) is tuple:
        num_workers_train = num_workers[0]
        num_workers_test = num_workers[1]
    else:
        num_workers_train = num_workers_test = num_workers

    if evaluate is True:
        n_train_samples = round(0.85 * len(dataset))
        n_val_samples = len(dataset) - n_train_samples
        dataset_train, dataset_val = random_split(
            dataset, lengths=[n_train_samples, n_val_samples], generator=torch.Generator().manual_seed(99))
        dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers_train)
        dl_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers_test)
    else:
        dl_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers_train)
        dl_val = None

    training_results = {
        'average_train_loss': [],
        'average_val_loss': [],
        'average_train_score': [],
        'average_val_score': []
    }

    print(f'Device: {device}.')

    clock_start = time.time()

    for epoch in range(n_epochs):
        model_on_device.train()

        average_train_loss, average_train_score = training_or_testing_epoch_(
            device, model_on_device, dl_train, score_fn, loss_fn, training=True, optimizer=optimizer)
        average_train_score = average_train_score.mean().item()
        training_results['average_train_loss'].append(average_train_loss)
        training_results['average_train_score'].append(average_train_score)

        if verbose is True:
            print(f'--- Epoch {epoch + 1}/{n_epochs} ---')
            print(f'average_train_loss: {average_train_loss}, average_train_score: {average_train_score}')
        
        if dl_val is None:
            continue
        
        model_on_device.eval()

        with torch.no_grad():
            average_val_loss, average_val_score = training_or_testing_epoch_(device, model_on_device, dl_val, score_fn, loss_fn)
        
        average_val_score = average_val_score.mean().item()
        training_results['average_val_loss'].append(average_val_loss)
        training_results['average_val_score'].append(average_val_score)

        if verbose is True:
            print(f'average_val_loss: {average_val_loss}, average_val_score: {average_val_score}')
        
        if lr_scheduler is not None:
            lr_scheduler.step()

    clock_end = time.time()

    model_on_device.eval()

    print(f'\nTraining completed in around {math.ceil((clock_end - clock_start) / 60)} minutes.')
    
    return training_results

# Function for testing model on dataset. Returns the average score along dataset's batches.
# ---
# device: device on which to load data and model ('cpu' for cpu).
# dataset: instance of class extending torch.utils.data.Dataset.
# score_fn: function to be used to evaluate a batch of predictions against the corresponding batch of targets.
def test_model(device, model, dataset, batch_size, score_fn):
    dl_test = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    model_on_device = model.to(device)
    model_on_device.eval()

    clock_start = time.time()

    with torch.no_grad():
        _, average_score = training_or_testing_epoch_(device, model_on_device, dl_test, score_fn)

    clock_end = time.time()

    print(f'Device: {device}.')
    print(f'\nInference completed in around {math.ceil(clock_end - clock_start)} seconds.')

    return average_score

# Plots training results, as returned from function train_model.
# ---
# results_dict: dictionary with keys average_train_loss, average_val_loss, average_train_score, average_val_score, 
#               each identifying a python list of losses/scores.
# plotsize: tuple representing the size (in inches) of the figure containing results_dict's plots.
# train_fmt, val_fmt: format string for plots of training and validation metrics respectively.
def plot_training_results(results_dict, plotsize, train_fmt='g', val_fmt='b'):
    assert(
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
    plt.plot(range(1, n_epochs + 1), results_dict['average_train_loss'], train_fmt, \
        range(1, n_epochs + 1), results_dict['average_val_loss'], val_fmt)
    plt.title('Average Loss')
    plt.xlabel(xlabel)
    plt.ylabel('Loss')
    plt.legend(legend)

    # average scores
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_epochs + 1), results_dict['average_train_score'], train_fmt, \
        range(1, n_epochs + 1), results_dict['average_val_score'], val_fmt)
    plt.title('Average Score')
    plt.xlabel(xlabel)
    plt.ylabel('Score')
    plt.legend(legend)

    plt.show()