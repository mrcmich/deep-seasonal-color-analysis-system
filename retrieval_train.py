import torch
from torch import nn
import open_clip
from models.dataset import DressCodeDataset
from utils import utils, model_names
from models import training_and_testing
from retrieval import training_and_testing_retrieval
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from slurm_scripts import slurm_config
from ray.tune.schedulers import ASHAScheduler
import os
from models import config


def main_worker(args):
    clip_model = args.clip_model
    pretrained = model_names.CLIP_MODELS_PRETRAINED[clip_model]
    model, preprocess, _ = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(clip_model)
    # Dataset & Dataloader
    dataset = DressCodeDataset(dataroot_path=args.dataroot,
                               preprocess=preprocess,
                               phase="train",
                               order=args.order)
    
    # if possible, exploit multiple GPUs
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
    plot_path = "retrieval/plots/training_test.png"
    evaluate = True
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.2)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    
    results = training_and_testing_retrieval.train_retrieval_model(device=device, model=model, tokenizer=tokenizer, dataset=dataset, n_epochs=n_epochs,
                                                                   batch_size=batch_size, loss_img=loss_img, loss_txt=loss_txt, optimizer=optimizer,
                                                                   evaluate=evaluate)
    
    training_and_testing.plot_training_results(results_dict=results, filepath=plot_path)
    


def main_worker_with_ray(args):
    clip_model = args.clip_model
    pretrained = model_names.CLIP_MODELS_PRETRAINED[clip_model]
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained)
    print(model.classification_head)
    tokenizer = open_clip.get_tokenizer(clip_model)
    # Dataset & Dataloader
    dataset = DressCodeDataset(dataroot_path=args.dataroot,
                               preprocess=preprocess,
                               phase=args.phase,
                               order=args.order)
    
    # if possible, exploit multiple GPUs
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
    # Ray Tune parameters
    cpus_per_trial = 0
    gpus_per_trial = torch.cuda.device_count()
    local_dir = config.RETRIEVAL_CHECKPOINTS_PATH
    num_samples = 1  # Number of times each combination is sampled (n_epochs are done per sample)
    
    evaluate = True
    n_epochs = args.n_epochs
    optimizer = torch.optim.Adam
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    
    max_report_frequency = 600
    if evaluate:
        metrics_columns = ["train_loss", "train_accuracy", "val_loss", "val_accuracy", "training_iteration"]
    else:
        metrics_columns = ["train_loss", "train_accuracy", "training_iteration"]
    reporter = CLIReporter(
        metric_columns=metrics_columns, max_report_frequency=max_report_frequency)
    
    tunerun_cfg = {
        'lr': 1e-5,
        'weight_decay': 0.2,
        'batch_size': args.batch_size,
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.RETRIEVAL_CHECKPOINTS_PATH) + '/'
    }
    
    tune.run(partial(training_and_testing_retrieval.train_retrieval_model_with_ray,
                         device=device, model=model, tokenizer=tokenizer, dataset=dataset, n_epochs=n_epochs,
                         loss_img=loss_img, loss_txt=loss_txt, optimizer=optimizer, evaluate=evaluate),
                 config=tunerun_cfg,
                 num_samples=num_samples,
                 resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
                 progress_reporter=reporter,
                 checkpoint_at_end=True,
                 checkpoint_freq=1,
                 local_dir=local_dir)


if __name__ == '__main__':
    # Get argparser configuration
    args = utils.parse_retrieval_arguments()
    # Call main worker
    ray = True
    if ray:
        main_worker_with_ray(args)
    else:
        main_worker(args)
