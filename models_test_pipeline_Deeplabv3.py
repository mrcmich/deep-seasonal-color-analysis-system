import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from models import dataset, training_and_testing
from models.cloud.Deeplabv3 import deeplabv3
from metrics_and_losses import metrics
from utils import segmentation_labels, utils
from models import config


if __name__ == "__main__":
    args = utils.parse_arguments_test_pipeline()
    dataset_path = config.DATASET_PATH
    
    # defining transforms
    tH, tW = 256, 256
    image_transform = T.Compose([T.Resize((tH, tW)), T.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)])
    target_transform = T.Compose([T.Resize((tH, tW))])

    # fetching dataset
    n_classes = len(segmentation_labels.labels)
    img_paths, label_paths = dataset.get_paths(dataset_path, file_name=config.DATASET_INDEX_NAME)
    _, X_test, _, Y_test = train_test_split(
        img_paths, label_paths, test_size=0.20, random_state=99, shuffle=True)
    test_dataset = dataset.MyDataset(X_test, Y_test, image_transform, target_transform)

    # training hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    
    model = deeplabv3.deeplabv3_resnet50(num_classes=n_classes)
    model.load_state_dict(torch.load(args.weights_path))
    
    # testing model on test dataset
    test_score_fn = metrics.batch_IoU
    label_names = list(segmentation_labels.labels.keys())
    batch_IoU = training_and_testing.test_model(device, model, test_dataset, batch_size, test_score_fn)
    batch_IoU_with_labels = { label: score for label, score in list(zip(label_names, batch_IoU.tolist())) }
    batch_mIoU = batch_IoU.mean().item()
    for label in batch_IoU_with_labels:
        print(f'batch_IoU_{label}: {batch_IoU_with_labels[label]}')
    print(f'batch_mIoU={batch_mIoU}')
