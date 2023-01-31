import torch
from torch import nn
from sklearn.model_selection import train_test_split
from models import dataset, training_and_testing
from models.local.FastSCNN.models import fast_scnn
from models.cloud.UNet import unet
from metrics_and_losses import metrics
from utils import segmentation_labels
import torchsummary
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from models import config
from slurm_scripts import slurm_config
import sys

def run_tuning(model_name):
    dataset_path = config.DATASET_PATH
    n_classes = len(segmentation_labels.labels)

    if model_name == 'fastscnn':
        model = fast_scnn.FastSCNN(n_classes) 
    elif model_name == 'unet':
        model = unet.UNet(out_channels=n_classes)

    model_config = slurm_config.SLURM_CFG_TUNING[model_name]
    image_transform = model_config['image_transform']
    target_transform = model_config['target_transform']

    # fetching dataset
    img_paths, label_paths = dataset.get_paths(dataset_path, file_name=config.DATASET_INDEX_NAME)
    X_train, _, Y_train, _ = train_test_split(img_paths, label_paths, test_size=0.20, random_state=99, shuffle=True)
    train_dataset = dataset.MyDataset(X_train, Y_train, image_transform, target_transform)

    # model parameters
    cfg = model_config['hpo_cfg']

    # training hyperparameters
    batch_size = cfg['batch_size']
    n_epochs = model_config['n_epochs']
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:  # if possible, exploit multiple GPUs
            model = nn.DataParallel(model)

    class_weights = torch.tensor(config.CLASS_WEIGHTS, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    score_fn = metrics.batch_mIoU

    # optimizer
    optimizer = model_config['optimizer']

    # scheduler
    lr_scheduler = model_config['lr_scheduler']

    # printing model summary
    tH, tW = model_config['input_size']
    model_summary = torchsummary.summary(model, input_data=(batch_size, 3, tH, tW), batch_dim=None, verbose=0)
    print(model_summary)

    # Ray Tune parameters
    cpus_per_trial = 0
    gpus_per_trial = torch.cuda.device_count()
    num_samples = 1  # Number of times each combination is sampled (n_epochs are done per sample)
    evaluate = False
    if evaluate:
        metrics_columns = ["train_loss", "train_score", "val_loss", "val_score", "training_iteration"]
    else:
        metrics_columns = ["train_loss", "train_score", "training_iteration"]
    reporter = CLIReporter(
            metric_columns=metrics_columns,
            max_report_frequency=600)

    # launching training
    results = tune.run(partial(training_and_testing.train_model_with_ray,
        device=device, model=model, dataset=train_dataset, n_epochs=n_epochs, score_fn=score_fn, 
        loss_fn=loss_fn, 
        optimizer=optimizer, lr_scheduler=lr_scheduler, num_workers=(0,0), evaluate=evaluate),
        config=cfg,
        num_samples=num_samples,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        progress_reporter=reporter,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        local_dir=model_config['local_dir'])

if __name__ == '__main__':
    args = sys.argv

    if len(args) == 1:
        sys.exit("Error: invalid syntax." +
            " Syntax: models_tuning.py <model_name>, where <model_name> is from list ['fastscnn', 'unet'].")

    if len(args) != 2:
        sys.exit("Error: invalid number of arguments. Pass a model from list ['fastscnn', 'unet'].")
    
    model_name = args[1]
    
    if model_name not in ['fastscnn', 'unet']:
        sys.exit("Error: invalid model_name." + 
            " Parameter model_name must be taken from list ['fastscnn', 'unet'].")

    run_tuning(model_name)

