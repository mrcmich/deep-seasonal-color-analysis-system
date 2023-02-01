import torch
from torch import nn
from sklearn.model_selection import train_test_split
from models import dataset, training_and_testing
from models.local.FastSCNN.models import fast_scnn
from models.cloud.UNet import unet
from metrics_and_losses import metrics
from utils import segmentation_labels, utils
import torchsummary
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from models import config
from slurm_scripts import slurm_config

def run_training_best(args):
    dataset_path = config.DATASET_PATH
    n_classes = len(segmentation_labels.labels)
    model_name = args.model_name
    evaluate = args.evaluate

    if model_name == 'fastscnn':
        model = fast_scnn.FastSCNN(n_classes) 
    elif model_name == 'unet':
        model = unet.UNet(out_channels=n_classes)

    model_config = slurm_config.SLURM_CFG_TRAINING_BEST[model_name]
    image_transform = model_config['image_transform']
    target_transform = model_config['target_transform']

    # fetching dataset
    img_paths, label_paths = dataset.get_paths(dataset_path, file_name=config.DATASET_INDEX_NAME)
    X_train, _, Y_train, _ = train_test_split(img_paths, label_paths, test_size=0.20, random_state=99, shuffle=True)
    train_dataset = dataset.CcncsaDataset(X_train, Y_train, image_transform, target_transform)

    # model parameters
    cfg = model_config['tunerun_cfg']

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

    # printing model summary
    tH, tW = model_config['input_size']
    model_summary = torchsummary.summary(model, input_data=(batch_size, 3, tH, tW), batch_dim=None, verbose=0)
    print(model_summary)

    # Ray Tune parameters
    cpus_per_trial = 0
    gpus_per_trial = torch.cuda.device_count()
    num_samples = 1  # Number of times each combination is sampled (n_epochs are done per sample)
    if evaluate:
        metrics_columns = ["train_loss", "train_score", "val_loss", "val_score", "training_iteration"]
    else:
        metrics_columns = ["train_loss", "train_score", "training_iteration"]
    reporter = CLIReporter(
            metric_columns=metrics_columns,
            max_report_frequency=600)

    # launching training
    results = tune.run(partial(training_and_testing.train_model,
                               device=device, model=model, dataset=train_dataset, n_epochs=n_epochs, score_fn=score_fn,
                               loss_fn=loss_fn,
                               optimizer=optimizer, num_workers=(0,0), evaluate=evaluate, class_weights=class_weights),
        config=cfg,
        num_samples=num_samples,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        progress_reporter=reporter,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        local_dir=model_config['local_dir'])

if __name__ == '__main__':
    args = utils.parse_arguments()
    
    run_training_best(args)

