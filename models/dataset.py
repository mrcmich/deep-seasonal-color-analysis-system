# --- Needed to import modules from other packages
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# ---

from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from .config import *
from utils import segmentation_labels, utils
from palette_classification import color_processing
import torchvision.transforms as T

# file_name: filename of .xml file associating each image to the corresponding label
def get_paths(dataset_path, file_name):
    tree = ET.parse(dataset_path + file_name)
    root = tree.getroot()
    img_paths = []
    label_paths = []
    for child in root:
        if child.tag == "srcimg":
            img_paths.append(dataset_path + child.attrib['name'])
        elif child.tag == "labelimg":
            label_paths.append(dataset_path + child.attrib['name'])

    assert(len(img_paths) == len(label_paths))
    return img_paths, label_paths


class MyDataset(Dataset):
    def __init__(self, img_paths, label_paths, image_transform, label_transform=None):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.image_transform = image_transform
        self.label_transform = label_transform if label_transform is not None else self.image_transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image, _ = utils.load_image(img_filename=self.img_paths[index])
        label, _ = utils.load_image(img_filename=self.label_paths[index])
        image = self.image_transform(image / 255)
        label = self.label_transform(label)
        label_masks = color_processing.compute_segmentation_masks(label, segmentation_labels.labels)
        
        return image, label_masks
