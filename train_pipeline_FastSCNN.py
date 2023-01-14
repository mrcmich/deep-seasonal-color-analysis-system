import torch
from torch import nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from models import dataset, training_and_testing
from models.local.FastSCNN.models import fast_scnn
from metrics_and_losses import metrics
from utils import segmentation_labels, utils
import torchsummary
from models.config import *


if __name__ == "__main__":
    args = utils.parse_arguments_train_pipeline()
    
    weights_path = 'models/weights/'
    plots_path = 'models/plots/'
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

    # training hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    n_epochs = args.n_epochs if args.evaluate else (args.n_epochs // 2)

    # model, loss, score function
    model_name = 'fast_scnn_ccncsa'
    model = fast_scnn.FastSCNN(n_classes)
    loss_fn = nn.CrossEntropyLoss()
    score_fn = metrics.batch_mIoU

    # optimizer
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_summary = torchsummary.summary(model, input_data=(batch_size, 3, tH, tW), batch_dim=None, verbose=0)
    print(model_summary)
    
    # training
    results = training_and_testing.train_model(
        device, model, train_dataset, batch_size, n_epochs, score_fn, loss_fn, optimizer, lr_scheduler=None, evaluate=args.evalute, verbose=True)
    
    if args.evaluate:
        # plotting training results
        training_and_testing.plot_training_results(results, plotsize=(20, 6), filepath=plots_path + model_name + "_training_curves.png")
    else:
        # saving final model's weights
        torch.save(model.state_dict(), weights_path + model_name + '.pth')
        
        # testing model on test dataset
        test_score_fn = metrics.batch_IoU
        label_names = list(segmentation_labels.labels.keys())
        batch_IoU = training_and_testing.test_model(device, model, test_dataset, batch_size, test_score_fn)
        batch_IoU_with_labels = { label: score for label, score in list(zip(label_names, batch_IoU.tolist())) }
        batch_mIoU = batch_IoU.mean().item()
        for label in batch_IoU_with_labels:
            print(f'batch_IoU_{label}: {batch_IoU_with_labels[label]}')
        print(f'batch_mIoU={batch_mIoU}')
