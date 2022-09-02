# --- Needed to import modules from other packages
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# ---

from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from .config import *
from utils import segmentation_labels
from palette_classification import color_processing

def get_paths(path):
    tree = ET.parse(ROOT_DIR + path)
    root = tree.getroot()
    img_paths = []
    label_paths = []
    for child in root:
        if child.tag == "srcimg":
            img_paths.append(ROOT_DIR + "headsegmentation_dataset_ccncsa/" + child.attrib['name'])
        elif child.tag == "labelimg":
            label_paths.append(ROOT_DIR + "headsegmentation_dataset_ccncsa/" + child.attrib['name'])

    assert(len(img_paths) == len(label_paths))
    return img_paths, label_paths


class MyDataset(Dataset):
    def __init__(self, img_paths, label_paths, transform):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert('RGB')
        label = Image.open(self.label_paths[index]).convert('RGB')
        image = self.transform(image).float()
        label = self.transform(label).float()
        label = color_processing.compute_segmentation_masks(label, segmentation_labels.labels)
        
        return image, label
