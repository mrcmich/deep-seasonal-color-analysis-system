import torch
from torch import nn
from torchvision import ops
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from models import dataset, training_and_testing
from models.cloud.UNet import unet
from metrics_and_losses import metrics
from utils import segmentation_labels, utils
import matplotlib.pyplot as plt
from palette_classification import color_processing
import torchsummary
from models.config import *
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import pprint


executing_on_colab = False

# local configuration
if executing_on_colab is False:
    weights_path = 'models/weights/'
    dataset_path = ROOT_DIR + 'headsegmentation_dataset_ccncsa/'  

# defining transforms
tH, tW = 256, 256
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # from ImageNet
image_transform = T.Compose([T.Resize((tH, tW)), T.Normalize(mean, std)])
target_transform = T.Compose([T.Resize((tH, tW))])

# fetching dataset
n_classes = len(segmentation_labels.labels)
img_paths, label_paths = dataset.get_paths(dataset_path, file_name='training.xml')
X_train, X_test, Y_train, Y_test = train_test_split(img_paths, label_paths, test_size=0.20, random_state=99, shuffle=True)
train_dataset = dataset.MyDataset(X_train, Y_train, image_transform, target_transform)
test_dataset = dataset.MyDataset(X_test, Y_test, image_transform, target_transform)

# setting up model and fixed (initially) hyperparameters
class_weights = torch.tensor([0.3762, 0.9946, 0.9974, 0.9855, 0.7569, 0.9140, 0.9968, 0.9936, 0.9989, 0.9893, 0.9968])

# === hyperparameters optimization (HPO) ===

model = unet.UNet(out_channels=n_classes)

# if possible, exploit multiple GPUs
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

# model parameters
config = {
    "lr": tune.grid_search([1e-4, 1e-2]),
    "batch_size": tune.grid_search([16, 32, 64]),
    "start_factor": tune.grid_search([0.3, 0.5]),
    "from_checkpoint": False,
    "checkpoint_dir": os.path.abspath("./models/hpo/UNet")
}
n_epochs = 5
score_fn = metrics.batch_mIoU
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Ray Tune parameters
cpus_per_trial = 0
gpus_per_trial = torch.cuda.device_count()
num_samples = 1  # Number of times each combination is sampled (n_epochs are done per sample)
scheduler = ASHAScheduler(grace_period=2)
reporter = CLIReporter(
        metric_columns=["loss", "score", "training_iteration"],
        max_report_frequency=300)

# launching HPO
hpo_results = tune.run(partial(training_and_testing.hpo,
    device=device, model=model, dataset=train_dataset, n_epochs=n_epochs, score_fn=score_fn, loss_fn=loss_fn, 
    optimizer=torch.optim.AdamW, lr_scheduler=torch.optim.lr_scheduler.LinearLR, num_workers=(0,0), evaluate=True),
    config=config,
    metric="loss", # This metric should be reported with `session.report()`
    mode="min",
    num_samples=num_samples,
    resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end=True,
    checkpoint_freq=1,
    local_dir="models/hpo/UNet")

# retrieve best results
# Get best trial
best_trial = hpo_results.best_trial
print(f"Best trial: {hpo_results.best_trial}")

# Get best trial's hyperparameters
pprint.pprint(f"Best trial configuration: {hpo_results.best_config}")

# Get best trial's log directory
print(f"Best trial log directory: {hpo_results.best_logdir}")

print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best trial final validation score: {}".format(
    best_trial.last_result["score"]))
