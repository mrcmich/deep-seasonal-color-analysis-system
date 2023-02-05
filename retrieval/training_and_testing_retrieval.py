import torch
from torch.utils.data import DataLoader
import time
import math


def test_retrieval_model(device, model, tokenizer, dataset, batch_size, num_workers=0):
    model = model.to(device)
    model.eval()
    
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    labels_list = dataset.get_labels()
    accuracy = 0
    
    clock_start = time.time()
    
    for inputs in dl:
        cloth_name = inputs["cloth_name"]
        cloth_img = inputs["cloth_img"].to(device)
        label = inputs["label"]
        label_idx = torch.zeros((len(label)), dtype=torch.int64)
        for i, l in enumerate(label):
            label_idx[i] = labels_list.index(l)
        
        text = tokenizer(labels_list).to(device)

        with torch.no_grad():
            image_features = model.encode_image(cloth_img)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            text_probs = (100.0 * text_features @ image_features.T).softmax(dim=-1)
            predictions = torch.argmax(text_probs, dim=0)
            accuracy += torch.sum(predictions.cpu() == label_idx).item()
    
    clock_end = time.time()
    
    print(f'Device: {device}.')
    print(f'Inference completed in around {math.ceil(clock_end - clock_start)} seconds.')
    
    accuracy /= len(dataset)
    return accuracy
