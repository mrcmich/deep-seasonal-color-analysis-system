import torch
from torch import nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from models import dataset, training_and_testing
from models.cloud.Deeplabv3 import deeplabv3
from metrics_and_losses import metrics
from utils import segmentation_labels, utils
import torchsummary
from models import config
from functools import partial
from ray import tune
from ray.tune import CLIReporter
import os

if __name__ == "__main__":
    args = utils.parse_arguments_train_pipeline()
    
    weights_path = config.WEIGHTS_PATH
    plots_path = config.PLOTS_PATH
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

    # training hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    n_epochs = args.n_epochs if args.evaluate else (args.n_epochs // 2)

    # model, loss, score function
    model_name = 'deeplabv3_ccncsa'
    model = deeplabv3.deeplabv3_resnet50(num_classes=n_classes)
    loss_fn = nn.CrossEntropyLoss()
    score_fn = metrics.batch_mIoU

    # optimizer
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_summary = torchsummary.summary(model, input_data=(batch_size, 3, tH, tW), batch_dim=None, verbose=0)
    print(model_summary)
    
    # training
    results = training_and_testing.train_model(
        device, model, train_dataset, batch_size, n_epochs, score_fn, loss_fn, optimizer, 
        lr_scheduler=None, evaluate=args.evalute, verbose=True)
    
    # model parameters
    cfg = {
        "lr": learning_rate,
        "batch_size": batch_size,
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.CHECKPOINTS_PATH + "Deeplabv3")
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

    results = tune.run(partial(training_and_testing.train_model_with_ray,
        device=device, model=model, dataset=train_dataset, n_epochs=n_epochs, score_fn=score_fn, loss_fn=loss_fn, 
        optimizer=optimizer, lr_scheduler=None, num_workers=(0,0), evaluate=args.evalute),
        config=cfg,
        num_samples=num_samples,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        progress_reporter=reporter,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        local_dir=config.CHECKPOINTS_PATH+"Deeplabv3")
    
    if args.evaluate:
        # plotting training results?
        pass
    else:
        # saving final model's weights
        torch.save(model.state_dict(), weights_path + model_name + '.pth')
