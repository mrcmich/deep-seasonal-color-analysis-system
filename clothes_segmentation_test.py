import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models.dataset import DressCodeDataset
from models import config
from retrieval import clothes_segmentation


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = DressCodeDataset(dataroot_path=config.DRESSCODE_PATH_ON_LAB_SERVER,
                               preprocess=T.Compose([T.ToTensor()]),
                               phase="test",
                               order="unpaired")

batch_size = 32
labels_list = dataset.get_labels()
dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

save_fig_path = "retrieval/segmented_clothes/"
count = 0
batch_to_check = 1
for inputs in dl:
    if count < batch_to_check:
        count += 1
        continue
    for idx in range(batch_size):
        dataroot = inputs["dataroot"][idx]
        cloth_name = inputs["cloth_name"][idx]
        img_path = dataroot + f"/images/{cloth_name}"
        segmentation_mask = clothes_segmentation.segment_img_cloth(img_path, save_fig_path)
            
    if count == batch_to_check:
        break
    count += 1
