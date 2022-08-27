from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import xml.etree.ElementTree as ET
from config import *
from utils import *


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
    def __init__(self, img_paths, label_paths):
        self.img_paths = img_paths
        self.label_paths = label_paths
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert('RGB')
        label = Image.open(self.label_paths[index]).convert('RGB')
        transform = T.Compose([T.PILToTensor(), T.Resize((200, 200))])
        image = transform(image).float()
        label = transform(label).float()
        label = from_DHW_to_HWD(label)
        label = compute_segmentation_masks(label, labels)
        label = from_HWD_to_DHW(label)
        
        return image, label
