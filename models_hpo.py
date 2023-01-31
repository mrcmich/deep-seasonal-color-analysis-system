import torch
from torch import nn
from sklearn.model_selection import train_test_split
from models import dataset, training_and_testing
from models.cloud.UNet import unet
from models.local.FastSCNN.models import fast_scnn
from metrics_and_losses import metrics
from utils import segmentation_labels
from models import config
from slurm_scripts import slurm_config
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import pprint
import sys

def run_hpo(model_name):
    dataset_path = config.DATASET_PATH
    
    if model_name == 'fastscnn':
        model = fast_scnn.FastSCNN(n_classes) 
    elif model_name == 'unet':
        model = unet.UNet(out_channels=n_classes)

    model_config = slurm_config.SLURM_CFG_HPO[model_name]
    image_transform = model_config['image_transform']
    target_transform = model_config['target_transform']

    # fetching dataset
    n_classes = len(segmentation_labels.labels)
    img_paths, label_paths = dataset.get_paths(dataset_path, file_name=config.DATASET_INDEX_NAME)
    X_train, _, Y_train, _ = train_test_split(
        img_paths, label_paths, test_size=0.20, random_state=99, shuffle=True)
    train_dataset = dataset.MyDataset(X_train, Y_train, image_transform, target_transform)
    
    # === hyperparameters optimization (HPO) ===

    # if possible, exploit multiple GPUs
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    # model parameters
    n_epochs = 5
    cfg = model_config['hpo_cfg']
    score_fn = metrics.batch_mIoU
    class_weights = torch.tensor(config.CLASS_WEIGHTS)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Ray Tune parameters
    cpus_per_trial = 0
    gpus_per_trial = torch.cuda.device_count()
    num_samples = 1  # Number of times each combination is sampled (n_epochs are done per sample)
    scheduler = ASHAScheduler(grace_period=2)
    evaluate = True
    if evaluate:
        metric = "val_loss"
        metrics_columns = ["train_loss", "train_score", "val_loss", "val_score", "training_iteration"]
    else:
        metric = "train_loss"
        metrics_columns = ["train_loss", "train_score", "training_iteration"]
    reporter = CLIReporter(
            metric_columns=metrics_columns,
            max_report_frequency=300)

    # launching HPO
    hpo_results = tune.run(partial(training_and_testing.train_model_with_ray,
        device=device, model=model, dataset=train_dataset, n_epochs=n_epochs, score_fn=score_fn, 
        loss_fn=loss_fn, 
        optimizer=model_config['optimizer'], lr_scheduler=model_config['lr_scheduler'], 
        num_workers=(0,0), evaluate=evaluate),
        config=cfg,
        metric=metric, # This metric should be reported with `session.report()`
        mode="min",
        num_samples=num_samples,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        local_dir=model_config['local_dir'])

    # retrieve best results
    # Get best trial
    best_trial = hpo_results.best_trial
    print(f"Best trial: {hpo_results.best_trial}")

    # Get best trial's hyperparameters
    pprint.pprint(f"Best trial configuration: {hpo_results.best_config}")

    # Get best trial's log directory
    print(f"Best trial log directory: {hpo_results.best_logdir}")

    if evaluate:
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["val_loss"]))
        print("Best trial final validation score: {}".format(
            best_trial.last_result["val_score"]))
    else:
        print("Best trial final training loss: {}".format(
            best_trial.last_result["train_loss"]))
        print("Best trial final training score: {}".format(
            best_trial.last_result["train_score"]))


if __name__ == '__main__':
    args = sys.argv

    if len(args) == 1:
        sys.exit("Error: invalid syntax." +
            " Syntax: models_hpo.py <model_name>, where <model_name> is from list ['fastscnn', 'unet'].")

    if len(args) != 2:
        sys.exit("Error: invalid number of arguments. Pass a model from list ['fastscnn', 'unet'].")
    
    model_name = args[1]
    
    if model_name not in ['fastscnn', 'unet']:
        sys.exit("Error: invalid model_name." + 
            " Parameter model_name must be taken from list ['fastscnn', 'unet'].")

    run_hpo(model_name)

