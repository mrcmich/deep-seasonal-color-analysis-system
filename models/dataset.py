# --- Needed to import modules from other packages
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# ---

import os
from typing import List, Tuple
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from utils import segmentation_labels
from palette_classification import color_processing
import torchvision.transforms as T


def get_paths(dataset_path, file_name):
    """
    .. inputs::
    file_name:  filename of .xml file associating each image to the corresponding label
    """
    tree = ET.parse(dataset_path + file_name)
    root = tree.getroot()
    img_paths = []
    label_paths = []
    for child in root:
        if child.tag == "srcimg":
            img_paths.append(dataset_path + child.attrib['name'])
        elif child.tag == "labelimg":
            label_paths.append(dataset_path + child.attrib['name'])

    assert (len(img_paths) == len(label_paths))
    return img_paths, label_paths


class CcncsaDataset(Dataset):
    def __init__(self, img_paths, label_paths, image_transform, label_transform=None):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.image_transform = image_transform
        self.label_transform = label_transform if label_transform is not None else self.image_transform
        self.pil_to_tensor = T.Compose([T.PILToTensor()])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert('RGB')
        label = Image.open(self.label_paths[index]).convert('RGB')
        image = self.image_transform(self.pil_to_tensor(image) / 255)
        label = self.label_transform(self.pil_to_tensor(label))
        label_masks = color_processing.compute_segmentation_masks(label, segmentation_labels.labels)

        return image, label_masks


class DressCodeDataset(Dataset):
    def __init__(self,
                 dataroot_path: str,
                 preprocess: T.Compose,
                 phase: str,
                 order: str = 'paired',
                 category: List[str] = ['dresses', 'upper_body', 'lower_body'],
                 size: Tuple[int, int] = (256, 192)):
        """
        Initialize the PyTroch Dataset Class
        :param dataroot_path: dataset root folder
        :type dataroot_path:  string
        :param preprocess: transform of clip model
        :type preprocess:  T.Compose
        :param phase: phase (train | test)
        :type phase: string
        :param order: setting (paired | unpaired)
        :type order: string
        :param category: clothing category (upper_body | lower_body | dresses)
        :type category: list(str)
        :param size: image size (height, width)
        :type size: tuple(int)
        """
        super(DressCodeDataset, self).__init__()
        self.dataroot = dataroot_path
        self.preprocess = preprocess
        self.phase = phase
        self.category = category
        self.height = size[0]
        self.width = size[1]
        self.transform = T.Compose([
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        cloth_names = []
        dataroot_names = []

        for c in category:
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = os.path.join(self.dataroot, c)
            if phase == 'train':
                filename = os.path.join(dataroot, f"{phase}_pairs.txt")
            else:
                filename = os.path.join(dataroot, f"{phase}_pairs_{order}.txt")
            with open(filename, 'r') as f:
                for line in f.readlines():
                    _, cloth_name = line.strip().split()
                    cloth_names.append(cloth_name)
                    dataroot_names.append(dataroot)

        self.cloth_names = cloth_names
        self.dataroot_names = dataroot_names

    def __getitem__(self, index):
        """
        For each index return the corresponding sample in the dataset
        :param index: data index
        :type index: int
        :return: dict containing dataset samples
        :rtype: dict
        """
        cloth_name = self.cloth_names[index]
        dataroot = self.dataroot_names[index]

        # Clothing image
        cloth_img = Image.open(os.path.join(dataroot, 'images', cloth_name))
        cloth_img = cloth_img.resize((self.width, self.height))
        cloth_img = self.preprocess(cloth_img)
        cloth_img = self.transform(cloth_img)   # [-1,1]

        label = "a cloth of type "
        if dataroot.split('/')[-1] == 'dresses':
            label += "dress"
        elif dataroot.split('/')[-1] == 'upper_body':
            label += "upper body"
        elif dataroot.split('/')[-1] == 'lower_body':
            label += "lower body"

        result = {
            'dataroot': dataroot,
            'cloth_name': cloth_name,  # for visualization
            'cloth_img': cloth_img,  # for input
            'label': label,
        }

        return result
    
    def get_labels(self):
        labels = []
        for c in self.category:
            label = "a cloth of type "
            if c == 'dresses':
                label += "dress"
            elif c == 'upper_body':
                label += "upper body"
            elif c == 'lower_body':
                label += "lower body"
            labels.append(label)
        
        return labels

    def __len__(self):
        return len(self.cloth_names)
