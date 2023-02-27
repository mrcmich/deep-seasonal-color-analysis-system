import torch
from torch.utils.data import DataLoader
import time
import math
import queue
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


def test_retrieval_model(device, model, tokenizer, dataset, batch_size, num_workers=0):
    """
    .. description::
    Function to evaluate accuracy of a CLIP model on the test set of DressCode Dataset.
    Both the classification task and retrieval task are evaluated: associate the correct string to one image;
    and given a string, retrieve the images that better matche with it.

    .. inputs::
    device:                     cpu or cuda.
    model:                      CLIP model to test.
    tokenizer:                  tokenizer relative to CLIP model that process the query strings.
    dataset:                    DressCode dataset object.

    .. outputs::
    Returns the two scores of accuracy: image2text and text2image.
    """
    
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
            values, indices = torch.topk(image_probs, image_probs.shape[-1], dim=-1)
            for cls in range(3):
                for val, idx in zip(values[cls, :], indices[cls, :]):
                    if val >= 0.5 and cls == label_idx[idx]:
                        text_image_accuracy += 1
                    if val < 0.5 and cls != label_idx[idx]:
                        text_image_accuracy += 1

    clock_end = time.time()
    
    image_text_accuracy /= len(dataset)
    text_image_accuracy /= 3 * len(dataset)
    
    print(f'Device: {device}.')
    print(f'Inference completed in around {math.ceil(clock_end - clock_start)} seconds.')
    
    return image_text_accuracy, text_image_accuracy


def retrieve_clothes(device, model, tokenizer, query, dataset, k=5, batch_size=32, save_img_path=None, num_workers=0, verbose=False):
    """
    .. description::
    Function that retrieves images from a dataset given a query string.

    .. inputs::
    device:                     cpu or cuda.
    model:                      CLIP model to use.
    tokenizer:                  tokenizer relative to CLIP model that process the query strings.
    query:                      class to search in the dataset (dresses | upper_body | lower_body).
    dataset:                    DressCode dataset object.
    k:                          how many images to retrieve.
    save_img_path:              path to save the images retrieved with their scores, default is None.

    .. outputs::
    Returns a list of paths of the retrieved images.
    If save_img_path is not none the retrieved images are saved with their scores.
    """
    
    model = model.to(device)
    model.eval()
    
    label = "a cloth of type " + query.replace("_", " ")
    
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    text = tokenizer(label).to(device)
    pq = queue.PriorityQueue()
    
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
            for idx, score in enumerate(image_probs[0, :]):
                score *= -1
                img_path = dataroot[idx] + "/images/" + cloth_name[idx]
                item = (score, img_path)
                pq.put(item)
    
    clock_end = time.time()
    
    if verbose:
        print(f'Device: {device}.')
        print(f'Retrieve process completed in around {math.ceil(clock_end - clock_start)} seconds.')
    
    retrieved_img_paths = []
    i = 0
    while not pq.empty():
        if i == k:
            break
        score, img_path = pq.get()
        retrieved_img_paths.append(img_path)
        if save_img_path is not None:
            score *= -1
            img = Image.open(img_path)
            plt.figure()
            plt.title(f"Score: {100 * score.item()}%")
            plt.imshow(img)
            plt.savefig(save_img_path + f"image_{i + 1}_{query}.png")
        i += 1
    
    return retrieved_img_paths
