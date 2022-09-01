import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import get_paths, MyDataset
from encoding.models import fcn
from config import *


path = "headsegmentation_dataset_ccncsa/training.xml"
img_paths, label_paths = get_paths(path)


BATCH_SIZE = 128
x_train, x_test, y_train, y_test = train_test_split(img_paths, label_paths, test_size=0.2, random_state=21, shuffle=True)
train_dataset = MyDataset(x_train, y_train)
test_dataset = MyDataset(x_test, y_test)

dl_train = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
dl_test = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)


learning_rate = 0.01
num_epochs = 10

model = fcn.FCN(nclass=11, backbone="resnet50")
loss_fun = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), learning_rate)


def eval(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in data_loader:
            pred = model(img)[0]
            pred = torch.where(pred < 0.5, 0, 1)
            acc = torch.sum((pred == label))
            correct += acc
            total += BATCH_SIZE * (200**2)
    return correct / total


for e in range(num_epochs):
    train_acc = eval(model, dl_train)
    test_acc = eval(model, dl_test)
    print(f"Epoch {e} train acc.: {train_acc:.3f} - test acc.: {test_acc:.3f}")
    
    for img, label in dl_train:
        opt.zero_grad()
        pred = model(img)[0]
        loss = loss_fun(pred, label)
        loss.backward()
        opt.step()
