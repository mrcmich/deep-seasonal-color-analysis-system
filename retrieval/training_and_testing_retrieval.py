import torch
from torch.utils.data import DataLoader
import time
import math
import queue
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


def test_retrieval_model(device, model, tokenizer, dataset, batch_size, num_workers=0):
    model = model.to(device)
    model.eval()
    
    labels_list = dataset.get_labels()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    
    image_text_accuracy = 0.0
    text_image_accuracy = 0.0
    clock_start = time.time()
    with torch.no_grad():
        for inputs in tqdm(dl):
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
            
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predictions = torch.argmax(text_probs, dim=-1)
            image_text_accuracy += torch.sum(predictions.cpu() == label_idx).item()
            
            image_probs = (100.0 * text_features @ image_features.T).softmax(dim=-1)
            predicted_image = torch.argmax(image_probs, dim=-1)
            ground_truth = label_idx[predicted_image]
            predictions = torch.arange(3, dtype=torch.int64)
            text_image_accuracy += torch.sum(predictions.cpu() == ground_truth).item()

    clock_end = time.time()
    
    image_text_accuracy /= len(dataset)
    text_image_accuracy /= 3 * len(dl)
    
    print(f'Device: {device}.')
    print(f'Inference completed in around {math.ceil(clock_end - clock_start)} seconds.')
    
    return image_text_accuracy, text_image_accuracy


def retrieve_clothes(device, model, tokenizer, query, k, dataset, batch_size, images_path, num_workers=0):
    model = model.to(device)
    model.eval()
    
    label = "a cloth of type " + query.replace("_", " ")
    
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    text = tokenizer(label).to(device)
    top_k = queue.PriorityQueue()
    
    clock_start = time.time()
    
    with torch.no_grad():
        for inputs in tqdm(dl):
            dataroot = inputs["dataroot"]
            cloth_name = inputs["cloth_name"]
            cloth_img = inputs["cloth_img"].to(device)

            image_features = model.encode_image(cloth_img)
            text_features = model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            image_probs = (100.0 * text_features @ image_features.T).softmax(dim=-1)
            best_of_batch_score, best_of_batch_idx = image_probs.topk(k, dim=-1)
            for score, idx in zip(best_of_batch_score[0], best_of_batch_idx[0]):
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
