import os
import numpy as np
import pandas as pd

from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class OPMDDataset(Dataset):

    def __init__(self, data_csv_path="data/opmd_dataset.csv", 
                image_dir="/ssd_scratch/cvit/chocolite/OPMD/images/", 
                mask_dir="/ssd_scratch/cvit/chocolite/OPMD/masks/", mode="train", transform=None, n_channels=3):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.data_csv = pd.read_csv(data_csv_path)

        self.mode = mode   
        self.transform = transform

        self.n_channels = n_channels

    def __len__(self):
        return self.data_csv.shape[0]

    def __getitem__(self, index):
        row = self.data_csv.iloc[index]

        image = Image.open(os.path.join(self.image_dir, f"{row['img_name'].split('.')[0]}.png"))
        mask = Image.open(os.path.join(self.mask_dir, f"{row['img_name'].split('.')[0]}_mask.png"))

        if self.n_channels == 1:
            image = np.array(image.convert("L"), dtype=np.float32)
        else:
            image = np.array(image.convert("RGB"), dtype=np.float32)
            
        mask = np.array(mask.convert("L"), dtype=np.float32)

        mask[mask != 0.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        print(torch.unique(image), torch.unique(mask))
        
        return image, mask
    

if __name__ == "__main__":
    dataset = OPMDDataset()
    print(len(dataset))
    print(dataset[0][0].shape, dataset[0][1].shape)

