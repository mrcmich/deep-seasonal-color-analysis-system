import torch
from torch import nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from models.local.FastSCNN.models import fast_scnn
from models.local.CGNet.model import CGNet
from models.local.LEDNet.models import lednet
from models.cloud.UNet import unet
from models.cloud.Deeplabv3 import deeplabv3
from models import dataset, training_and_testing
from metrics_and_losses import metrics
from utils import segmentation_labels, utils
import torchsummary
from models import config
from functools import partial
from ray import tune
from ray.tune import CLIReporter
import os

MODEL_DICT = {
    "fastscnn": "FastSCNN"
}

def run_training_demo(args):
    dataset_path = config.DATASET_PATH

    # defining transforms
    tH, tW = 256, 256
    image_transform = T.Compose([T.Resize((tH, tW)), T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)])
    target_transform = T.Compose([T.Resize((tH, tW))])

    # fetching dataset
    n_classes = len(segmentation_labels.labels)
    img_paths, label_paths = dataset.get_paths(dataset_path, file_name=config.DATASET_INDEX_NAME)
    X_train, _, Y_train, _ = train_test_split(
        img_paths, label_paths, test_size=0.20, random_state=99, shuffle=True)
    train_dataset = dataset.CcncsaDataset(X_train, Y_train, image_transform, target_transform)

    # training hyperparameters
    # if possible, exploit multiple GPUs
    batch_size = 32
    n_epochs = args.n_epochs if args.evaluate else (args.n_epochs // 2)

    # model, loss, score function
    class_weights = torch.tensor(config.CLASS_WEIGHTS, device=device)
    
    if args.model_name == "fastscnn":
        model = fast_scnn.FastSCNN(n_classes)
    elif args.model_name == "cgnet":
        model = CGNet.Context_Guided_Network(classes=n_classes)
    elif args.model_name == "lednet":
        model = lednet.LEDNet(num_classes=n_classes, output_size=(tH, tW))
    elif args.model_name == "unet":
        model = unet.UNet(out_channels=n_classes)
    elif args.model_name == "deeplab":
        model = deeplabv3.deeplabv3_resnet50(num_classes=n_classes)
    else:
        raise Exception("model not supported.")
    
    loss_fn = nn.CrossEntropyLoss()
    score_fn = metrics.batch_mIoU
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    # optimizer
    learning_rate = 0.01
    optimizer = torch.optim.Adam

    model_summary = torchsummary.summary(model, input_data=(batch_size, 3, tH, tW), batch_dim=None, verbose=0)
    print(model_summary)
    
    # model parameters
    cfg = {
        "lr": learning_rate,
        "batch_size": batch_size,
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.DEMO_PATH + args.model_name)
    }

    # Ray Tune parameters
    cpus_per_trial = 0
    gpus_per_trial = torch.cuda.device_count()
    num_samples = 1  # Number of times each combination is sampled (n_epochs are done per sample)
    evaluate = args.evaluate
    if evaluate:
        metrics_columns = ["train_loss", "train_score", "val_loss", "val_score", "training_iteration"]
    else:
        metrics_columns = ["train_loss", "train_score", "training_iteration"]
    reporter = CLIReporter(
            metric_columns=metrics_columns,
            max_report_frequency=600)

    results = tune.run(partial(training_and_testing.train_model,
                               device=device, model=model, dataset=train_dataset, n_epochs=n_epochs, score_fn=score_fn, loss_fn=loss_fn,
                               optimizer=optimizer, num_workers=(0,0), evaluate=evaluate, class_weights=class_weights),
        config=cfg,
        num_samples=num_samples,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        progress_reporter=reporter,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        local_dir=config.CHECKPOINTS_PATH+args.model_name)


if __name__ == "__main__":
    args = utils.parse_arguments_demo()
    run_training_demo(args)
