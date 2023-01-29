import torch
from torch import nn, optim
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from models import dataset, training_and_testing
from models.cloud.UNet import unet
from metrics_and_losses import metrics
from utils import segmentation_labels, custom_transforms
import torchsummary
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from models.config import *


# local configuration
dataset_path = ROOT_DIR + 'headsegmentation_dataset_ccncsa/'

# defining transforms
tH, tW = 256, 256
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # from ImageNet
bilateral_filter = custom_transforms.BilateralFilter(sigma_color=50, sigma_space=100, diameter=7)

image_transform = T.Compose([
    T.Resize((tH, tW)), 
    bilateral_filter,
    T.Normalize(mean, std)])

target_transform = T.Compose([T.Resize((tH, tW))])

# fetching dataset
n_classes = len(segmentation_labels.labels)
img_paths, label_paths = dataset.get_paths(dataset_path, file_name='training.xml')
X_train, X_test, Y_train, Y_test = train_test_split(img_paths, label_paths, test_size=0.20, random_state=99, shuffle=True)
train_dataset = dataset.MyDataset(X_train, Y_train, image_transform, target_transform)
test_dataset = dataset.MyDataset(X_test, Y_test, image_transform, target_transform)

# training hyperparameters
batch_size = 16
n_epochs = 30

# model, loss, score function
model = unet.UNet(out_channels=n_classes)
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
learning_rate = 0.0001
optimizer = optim.AdamW

# scheduler
start_factor = 0.5
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
    "checkpoint_dir": os.path.abspath("./models/tuned_training/UNet")
}

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
    device=device, model=model, dataset=train_dataset, n_epochs=n_epochs, score_fn=score_fn, loss_fn=loss_fn, 
    optimizer=optimizer, lr_scheduler=lr_scheduler, num_workers=(0,0), evaluate=evaluate),
    config=config,
    num_samples=num_samples,
    resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
    progress_reporter=reporter,
    checkpoint_at_end=True,
    checkpoint_freq=1,
    local_dir="models/tuned_training/UNet")