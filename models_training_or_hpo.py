import torch
from torch import nn
from sklearn.model_selection import train_test_split
from models.local.FastSCNN.models import fast_scnn
from models.local.CGNet.model import CGNet
from models.local.LEDNet.models import lednet
from models.cloud.UNet import unet
from models.cloud.Deeplabv3 import deeplabv3
from models import dataset, training_and_testing
from metrics_and_losses import metrics
from utils import segmentation_labels, utils, model_names
import torchsummary
from models import config
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from slurm_scripts import slurm_config
from ray.tune.schedulers import ASHAScheduler
import pprint


def run_training_or_hpo(args):
    # fetching arguments
    cfg_name = args.config
    model_name = args.model_name
    evaluate = args.evaluate
    n_epochs = args.n_epochs

    # fetching configurations for model and tune.run
    cfg = slurm_config.configurations[cfg_name]
    model_cfg = cfg[model_name]
    model_cfg['local_dir'] = model_cfg['local_dir'] + model_names.MODEL_NAMES[model_name]
    tunerun_cfg = model_cfg['tunerun_cfg']
    tunerun_cfg['checkpoint_dir'] = tunerun_cfg['checkpoint_dir'] + model_names.MODEL_NAMES[model_name]
    is_hpo_cfg = model_cfg['hpo_cfg']  # True if model_cfg is a hpo configuration
    
    if cfg_name == "best":
        subfolder = "/validation" if evaluate is True else "/complete"
        tunerun_cfg['checkpoint_dir'] += subfolder
        model_cfg['local_dir'] += subfolder

    # defining transforms
    tH, tW = model_cfg['input_size']
    image_transform = model_cfg['image_transform']
    target_transform = model_cfg['target_transform']

    # fetching dataset
    n_classes = len(segmentation_labels.labels)
    dataset_path = config.DATASET_PATH
    img_paths, label_paths = dataset.get_paths(dataset_path, file_name=config.DATASET_INDEX_NAME)
    X_train, _, Y_train, _ = train_test_split(
        img_paths, label_paths, test_size=0.20, random_state=99, shuffle=True)
    train_dataset = dataset.CcncsaDataset(X_train, Y_train, image_transform, target_transform)

    # instantiating model
    if model_name == "fastscnn":
        model = fast_scnn.FastSCNN(n_classes)
    elif model_name == "cgnet":
        model = CGNet.Context_Guided_Network(classes=n_classes)
    elif model_name == "lednet":
        model = lednet.LEDNet(num_classes=n_classes, output_size=(tH, tW))
    elif model_name == "unet":
        model = unet.UNet(out_channels=n_classes)
    elif model_name == "deeplab":
        model = deeplabv3.deeplabv3_resnet50(num_classes=n_classes)
    else:
        raise Exception("model not supported.")

    # if possible, exploit multiple GPUs
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    # defining loss, optimizer and score function
    class_weights = torch.tensor(config.CLASS_WEIGHTS, device=device)
    is_loss_weighted = model_cfg['weighted_loss']
    loss_fn = nn.CrossEntropyLoss(class_weights) if is_loss_weighted else nn.CrossEntropyLoss()
    optimizer = model_cfg['optimizer']
    score_fn = metrics.batch_mIoU

    # printing summary of model
    if not is_hpo_cfg:
        batch_size = tunerun_cfg['batch_size']
        model_summary = torchsummary.summary(model, input_data=(batch_size, 3, tH, tW), batch_dim=None, verbose=0)
        print(model_summary)

    # Ray Tune parameters
    cpus_per_trial = 0
    gpus_per_trial = torch.cuda.device_count()
    local_dir = model_cfg['local_dir']
    num_samples = 1  # Number of times each combination is sampled (n_epochs are done per sample)
    
    # setting up the scheduler for early stopping of bad performing combinations
    if is_hpo_cfg:
        scheduler = ASHAScheduler(grace_period=10)  # set grace_period = num_epochs to avoid early stopping

    # setting up the reporter for printing metrics
    max_report_frequency = 600
    if evaluate:
        metric = "val_loss"
        metrics_columns = ["train_loss", "train_score", "val_loss", "val_score", "training_iteration"]
    else:
        metric = "train_loss"
        metrics_columns = ["train_loss", "train_score", "training_iteration"]
    reporter = CLIReporter(
        metric_columns=metrics_columns, max_report_frequency=max_report_frequency)

    if not is_hpo_cfg:
        # launching training
        tune.run(partial(training_and_testing.train_model,
                         device=device, model=model, dataset=train_dataset, n_epochs=n_epochs,
                         score_fn=score_fn, loss_fn=loss_fn, optimizer=optimizer, num_workers=(0, 0),
                         evaluate=evaluate, class_weights=class_weights),
                 config=tunerun_cfg,
                 num_samples=num_samples,
                 resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
                 progress_reporter=reporter,
                 checkpoint_at_end=True,
                 checkpoint_freq=1,
                 local_dir=local_dir)
    else:
        # launching HPO
        results = tune.run(partial(training_and_testing.train_model,
                                   device=device, model=model, dataset=train_dataset, n_epochs=n_epochs,
                                   score_fn=score_fn, loss_fn=loss_fn, optimizer=optimizer, num_workers=(0, 0),
                                   evaluate=evaluate, class_weights=class_weights),
                           config=tunerun_cfg,
                           metric=metric,  # This metric should be reported with `session.report()`
                           mode="min",
                           num_samples=num_samples,
                           resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
                           scheduler=scheduler,
                           progress_reporter=reporter,
                           checkpoint_at_end=True,
                           checkpoint_freq=1,
                           local_dir=local_dir)

        # retrieve best results
        # Get best trial
        best_trial = results.best_trial
        print(f"Best trial: {results.best_trial}")

        # Get best trial's hyperparameters
        pprint.pprint(f"Best trial configuration: {results.best_config}")

        # Get best trial's log directory
        print(f"Best trial log directory: {results.best_logdir}")

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


if __name__ == "__main__":
    args = utils.parse_training_or_hpo_arguments()
    run_training_or_hpo(args)
