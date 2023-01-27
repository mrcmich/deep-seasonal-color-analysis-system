import torch
from torch import nn, optim
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from models import dataset, training_and_testing
from models.local.FastSCNN.models import fast_scnn
from metrics_and_losses import metrics
from utils import segmentation_labels, utils, custom_transforms
import matplotlib.pyplot as plt
from palette_classification import color_processing
import torchsummary
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import pprint
from models.config import *


# local configuration
weights_path = 'models/weights/'
dataset_path = ROOT_DIR + 'headsegmentation_dataset_ccncsa/'

# === defining transforms ===
tH, tW = 512, 512
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # from ImageNet
bilateral_filter = custom_transforms.BilateralFilter(sigma_color=50, sigma_space=100, diameter=7)
center_crop = custom_transforms.PartiallyDeterministicCenterCrop(p=0.5)

train_image_transform = T.Compose([
    center_crop,
    T.ColorJitter(brightness=0.25, contrast=0.25), 
    T.Resize((tH, tW)), 
    bilateral_filter,
    T.Normalize(mean, std)])

train_target_transform = T.Compose([
    center_crop,
    T.Resize((tH, tW))])

test_image_transform = T.Compose([
    T.Resize((tH, tW)), 
    bilateral_filter,
    T.Normalize(mean, std)])

test_target_transform = T.Compose([T.Resize((tH, tW))])

# fetching dataset
n_classes = len(segmentation_labels.labels)
img_paths, label_paths = dataset.get_paths(dataset_path, file_name='training.xml')
X_train, X_test, Y_train, Y_test = train_test_split(img_paths, label_paths, test_size=0.20, random_state=99, shuffle=True)
train_dataset = dataset.MyDataset(X_train, Y_train, train_image_transform, train_target_transform)
test_dataset = dataset.MyDataset(X_test, Y_test, test_image_transform, test_target_transform)

# training hyperparameters
batch_size = 32
n_epochs = 30

# model, loss, score function
model_name = 'fast_scnn_ccncsa_tuned'
model = fast_scnn.FastSCNN(n_classes)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:  # if possible, exploit multiple GPUs
        model = nn.DataParallel(model)

class_weights = torch.tensor(
    [0.3762, 0.9946, 0.9974, 0.9855, 0.7569, 0.9140, 0.9968, 0.9936, 0.9989, 0.9893, 0.9968], device=device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
score_fn = metrics.batch_mIoU

# optimizer
learning_rate = 0.01
optimizer = optim.Adam

# scheduler
start_factor = 0.3
lr_scheduler = optim.lr_scheduler.LinearLR

# printing model summary
model_summary = torchsummary.summary(model, input_data=(batch_size, 3, tH, tW), batch_dim=None, verbose=0)
print(model_summary)

# model parameters
config = {
    "lr": learning_rate,
    "batch_size": batch_size,
    "start_factor": start_factor,
    "from_checkpoint": False,
    "checkpoint_dir": os.path.abspath("./models/tuned_training/FastFCNN")
}

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
    optimizer=optimizer, lr_scheduler=lr_scheduler, num_workers=(0,0), evaluate=False),
    config=config,
    metric="loss", # This metric should be reported with `session.report()`
    mode="min",
    num_samples=num_samples,
    resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end=True,
    checkpoint_freq=1,
    local_dir="models/tuned_training/FastFCNN")
