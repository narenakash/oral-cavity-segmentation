import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm

from monai.transforms import Activations, AsDiscrete, Compose

from models.unet import UNet

from utils import get_config, set_seed


config = get_config("config.yaml")
set_seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

post_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
])


# dataloader for all images in two folders
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, images, transform=None):
        self.path = path
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.path, self.images[idx]))

        # enhance contrast of the image by find percentiles, scale values according to the percentiles
        # p2, p98 = np.percentile(image, (2, 98))
        # image = Image.fromarray(np.uint8(np.clip(image, p2, p98)))

        # resize PIL image to 256x256
        image = image.resize((256, 256))

        image = np.array(image.convert("RGB"), dtype=np.float32)

        # if self.transform:
        #     image = self.transform(image)

        image = transforms.ToTensor()(image)
        return image
    

fold = 1
n_channels = 3
checkpoint_path = '/ssd_scratch/cvit/chocolite/OPMD/weights/unet_fold_1_dataaug_0_nchannel_3_lr_0.0001_bs_16_epochs_100_29-01-2024_22-16-26/model_80.pth'

# fold 1, RGB: unet_fold_1_dataaug_0_nchannel_3_lr_0.0001_bs_16_epochs_100_29-01-2024_22-16-26/model_80.pth


model = UNet(n_channels=n_channels, n_classes=1).to(device)
model = nn.DataParallel(model, [0,1,2,3])

loaded_dict = torch.load(checkpoint_path)['model_state_dict']

# prefix = 'module.'
# n_clip = len(prefix)
# adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
#                 if k.startswith(prefix)}

# model.load_state_dict(adapted_dict)

model.load_state_dict(loaded_dict)
model.eval()

for subfolder in ['train/non-suspicious', 'val/suspicious', 'val/non-suspicious', 'test/suspicious', 'test/non-suspicious']:
    path = '/ssd_scratch/cvit/chocolite/OPMD-Classification/' + subfolder
    images = os.listdir(path)

    save_dir = f'/ssd_scratch/cvit/chocolite/OPMD-SegClassification-F{fold}-N{n_channels}-ContrastEn/' + subfolder

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    dataset = Dataset(path, images, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, image in tqdm(enumerate(dataloader), total=len(dataloader)):
        image = image.to(device)

        with torch.no_grad():
            output = model(image)
            output = post_transform(output)

            output = output.squeeze().cpu().numpy()
            output = output.astype(np.uint8)
            output = np.expand_dims(output, axis=2)

            image = image.squeeze().cpu().numpy()
            image = image.transpose(1,2,0)
            image = image.astype(np.float32)

            output = np.array(output)
            image = np.array(image)

            segmented_image = output * image

            segmented_image = Image.fromarray(segmented_image.astype(np.uint8))
            segmented_image.save(os.path.join(save_dir, images[i])) 