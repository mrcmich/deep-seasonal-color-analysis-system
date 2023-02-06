import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from ray.air import session
import time
import math
import os
import queue
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


def training_or_testing_epoch_retrieval_(device, model, tokenizer, dl, len_dataset, labels_list, loss_img=None, loss_txt=None,
                                         optimizer=None, training=True):
    if training:
        assert (optimizer is not None and loss_img is not None and loss_txt is not None)
    
    cum_accuracy = 0.0
    cum_loss = 0.0
    
    for inputs in dl:
        cloth_img = inputs["cloth_img"].to(device)
        label = inputs["label"]
        label_idx = torch.zeros((len(label)), dtype=torch.int64)
        for i, l in enumerate(label):
            label_idx[i] = labels_list.index(l)
        
        text = tokenizer(label) if training else tokenizer(labels_list)
        text = text.to(device)
        ground_truth = torch.arange(len(label), dtype=torch.long, device=device)
        
        if training:
            optimizer.zero_grad()
            image_features, text_features, _ = model(cloth_img, text)
        else:
            image_features = model.encode_image(cloth_img)
            text_features = model.encode_text(text)
        
        logits_per_image = 100.0 * image_features.clone() @ text_features.clone().T
        logits_per_text = 100.0 * text_features.clone() @ image_features.clone().T
        
        if loss_img is not None and loss_txt is not None:
            tot_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            cum_loss += tot_loss.item()

        with torch.no_grad():
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predictions = torch.argmax(image_probs, dim=1)
            cum_accuracy += torch.sum(predictions.cpu() == label_idx).item()
        
        if training:
            tot_loss.backward()
            optimizer.step()
    
    accuracy = cum_accuracy / len_dataset
    loss = cum_loss / len_dataset
    return accuracy, loss
    
def train_retrieval_model(device, model, tokenizer, dataset, n_epochs, batch_size, loss_img, loss_txt, optimizer, evaluate=False, num_workers=0):
    model = model.to(device)
    labels_list = dataset.get_labels()
    torch.autograd.set_detect_anomaly(True)
    
    if evaluate:
        n_train_samples = round(0.85 * len(dataset))
        n_val_samples = len(dataset) - n_train_samples
        dataset_train, dataset_val = random_split(
            dataset, lengths=[n_train_samples, n_val_samples], generator=torch.Generator().manual_seed(99))
        dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers)
        dl_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)
    else:
        dl_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers)
        dl_val = None
    
    average_train_loss = []
    average_val_loss = []
    average_train_score = []
    average_val_score = []
    for e in range(n_epochs):
        model.train()
        train_accuracy, train_loss = training_or_testing_epoch_retrieval_(device=device, model=model, tokenizer=tokenizer, dl=dl_train,
                                                                          len_dataset=len(dataset_train), labels_list=labels_list, loss_img=loss_img,
                                                                          loss_txt=loss_txt, optimizer=optimizer, training=True)
        average_train_score.append(train_accuracy)
        average_train_loss.append(train_loss)

        if dl_val is not None:
            model.eval()
            with torch.no_grad():
                val_loss, val_accuracy = training_or_testing_epoch_retrieval_(device=device, model=model, tokenizer=tokenizer, dl=dl_val,
                                                                              len_dataset=len(dataset_val), labels_list=labels_list, loss_img=loss_img,
                                                                              loss_txt=loss_txt, training=False)
                average_val_score.append(val_accuracy)
                average_val_loss.append(val_loss)
        
        print(f"--- Epoch {e + 1} ---")
        print(f"train loss: {train_loss} - train accuracy: {train_accuracy}")
        if dl_val is not None:
            print(f"val loss: {val_loss} - val accuracy: {val_accuracy}")
    
    results = {
        'average_train_loss': average_train_loss,
        'average_train_score': average_train_score,
        'average_val_loss': average_val_loss,
        'average_val_score': average_val_score
    }

    model.eval()
    return results

def train_retrieval_model_with_ray(config, device, model, tokenizer, dataset, n_epochs, loss_img, loss_txt, optimizer, evaluate=False, num_workers=0):
    model = model.to(device)
    labels_list = dataset.get_labels()
    
    batch_size = config["batch_size"]
    optimizer = optimizer(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    # start from a checkpoint
    if config["from_checkpoint"]:
        model_state, optimizer_state = torch.load(os.path.join(config["checkpoint_dir"], "checkpoint.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    if evaluate:
        n_train_samples = round(0.85 * len(dataset))
        n_val_samples = len(dataset) - n_train_samples
        dataset_train, dataset_val = random_split(
            dataset, lengths=[n_train_samples, n_val_samples], generator=torch.Generator().manual_seed(99))
        dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers)
        dl_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers)
    else:
        dl_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers)
        dl_val = None
    
    
    for _ in range(n_epochs):
        model.train()
        train_accuracy, train_loss = training_or_testing_epoch_retrieval_(device=device, model=model, tokenizer=tokenizer, dl=dl_train,
                                                                          len_dataset=len(dataset_train), labels_list=labels_list, loss_img=loss_img,
                                                                          loss_txt=loss_txt, optimizer=optimizer, training=True)

        if dl_val is not None:
            model.eval()
            with torch.no_grad():
                val_loss, val_accuracy = training_or_testing_epoch_retrieval_(device=device, model=model, tokenizer=tokenizer, dl=dl_val,
                                                                              len_dataset=len(dataset_val), labels_list=labels_list, loss_img=loss_img,
                                                                              loss_txt=loss_txt, training=False)

        torch.save(
            (model.state_dict(), optimizer.state_dict()),
            os.path.join(config["checkpoint_dir"], "checkpoint.pt"))

        # report metrics to Ray Tune
        if evaluate:
            session.report({"train_loss": train_loss, "train_accuracy": train_accuracy,
                            "val_loss": val_loss, "val_accuracy": val_accuracy})
        else:
            session.report({"train_loss": train_loss, "train_accuracy": train_accuracy})

    model.eval()


def test_retrieval_model(device, model, tokenizer, dataset, batch_size, num_workers=0):
    model = model.to(device)
    model.eval()
    
    labels_list = dataset.get_labels()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    accuracy = 0.0
    clock_start = time.time()
    with torch.no_grad():
        for inputs in dl:
            cloth_img = inputs["cloth_img"].to(device)
            label = inputs["label"]
            label_idx = torch.zeros((len(label)), dtype=torch.int64)
            for i, l in enumerate(label):
                label_idx[i] = labels_list.index(l)

            text = tokenizer(labels_list).to(device)
            image_features = model.encode_image(cloth_img)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predictions = torch.argmax(image_probs, dim=1)
            accuracy += torch.sum(predictions.cpu() == label_idx).item()

    clock_end = time.time()
    
    accuracy /= len(dataset)
    
    print(f'Device: {device}.')
    print(f'Inference completed in around {math.ceil(clock_end - clock_start)} seconds.')
    
    return accuracy


def retrieve_clothes(device, model, tokenizer, query, k, dataset, batch_size, images_path, num_workers=0):
    model = model.to(device)
    model.eval()
    
    label = "a cloth of type " + query
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    text = tokenizer(label).to(device)
    top_k = queue.PriorityQueue()
    
    clock_start = time.time()
    
    with torch.no_grad():
        for i, inputs in tqdm(enumerate(dl)):
            dataroot = inputs["dataroot"]
            cloth_name = inputs["cloth_name"]
            cloth_img = inputs["cloth_img"].to(device)

            image_features = model.encode_image(cloth_img)
            text_features = model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            image_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            best_of_batch_score, best_of_batch_idx = image_probs.topk(k, dim=0)
            for score, idx in zip(best_of_batch_score, best_of_batch_idx):
                score *= -1
                img_path = dataroot[idx] + "/images/" + cloth_name[idx]
                item = (score, img_path)
                top_k.put(item)
    
    clock_end = time.time()
    
    print(f'Device: {device}.')
    print(f'Retrieve process completed in around {math.ceil(clock_end - clock_start)} seconds.')
    
    for i in range(1, k + 1):
        score, img_path = top_k.get()
        score *= -1
        img = Image.open(img_path)
        plt.figure()
        plt.title(f"Score: {100 * score.item()}%")
        plt.imshow(img)
        plt.savefig(images_path + f"image_{i}_{query}.png")
