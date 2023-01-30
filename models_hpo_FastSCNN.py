import torch
from torch import nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from models import dataset, training_and_testing
from models.local.FastSCNN.models import fast_scnn
from metrics_and_losses import metrics
from utils import segmentation_labels
from models import config
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import pprint
import os


weights_path = config.WEIGHTS_PATH
dataset_path = config.DATASET_PATH

# defining transforms
tH, tW = 256, 256
image_transform = T.Compose([T.Resize((tH, tW)), T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)])
target_transform = T.Compose([T.Resize((tH, tW))])

# fetching dataset
n_classes = len(segmentation_labels.labels)
img_paths, label_paths = dataset.get_paths(dataset_path, file_name=config.DATASET_INDEX_NAME)
X_train, X_test, Y_train, Y_test = train_test_split(
    img_paths, label_paths, test_size=0.20, random_state=99, shuffle=True)
train_dataset = dataset.MyDataset(X_train, Y_train, image_transform, target_transform)
test_dataset = dataset.MyDataset(X_test, Y_test, image_transform, target_transform)

# setting up model and fixed (initially) hyperparameters
class_weights = torch.tensor(config.CLASS_WEIGHTS)

# === hyperparameters optimization (HPO) ===

model = fast_scnn.FastSCNN(n_classes)

# if possible, exploit multiple GPUs
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

# model parameters
cfg = {
    "lr": tune.grid_search([1e-4, 1e-2]),
    "batch_size": tune.grid_search([16, 32, 64]),
    "start_factor": tune.grid_search([0.3, 0.5]),
    "from_checkpoint": False,
    "checkpoint_dir": os.path.abspath("./" + config.HPO_PATH + "FastSCNN")
}
n_epochs = 5
score_fn = metrics.batch_mIoU
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
    device=device, model=model, dataset=train_dataset, n_epochs=n_epochs, score_fn=score_fn, loss_fn=loss_fn, 
    optimizer=torch.optim.Adam, lr_scheduler=torch.optim.lr_scheduler.LinearLR, num_workers=(0,0), evaluate=evaluate),
    config=cfg,
    metric=metric, # This metric should be reported with `session.report()`
    mode="min",
    num_samples=num_samples,
    resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end=True,
    checkpoint_freq=1,
    local_dir=config.HPO_PATH+"FastSCNN")

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
