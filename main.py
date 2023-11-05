import wandb

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models.unet import UNet
# from losses import DiceCELoss
from dataset import OPMDDataset
from utils import get_config, set_seed

from train import train
# from test import test

from monai.losses import DiceCELoss

config = get_config("config.yaml")



def main():

    set_seed(config["seed"])

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    train_dataset = OPMDDataset(data_csv_path=config["dataset"]["train_csv_path"], image_dir=config["dataset"]["train_img_dir"], 
                                mask_dir=config["dataset"]["train_mask_dir"], mode="train", transform=None)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

    val_dataset = OPMDDataset(data_csv_path=config["dataset"]["val_csv_path"], image_dir=config["dataset"]["val_img_dir"],
                                mask_dir=config["dataset"]["val_mask_dir"], mode="val", transform=None)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    test_dataset = OPMDDataset(data_csv_path=config["dataset"]["test_csv_path"], image_dir=config["dataset"]["test_img_dir"],
                                mask_dir=config["dataset"]["test_mask_dir"], mode="test", transform=None)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    # model
    model = UNet(n_channels=config["model"]["n_channels"], n_classes=config["model"]["n_classes"]).to(device)
    model = nn.DataParallel(model)

    # loss and optimizer
    criterion = DiceCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["init_lr"])


    # train, eval and test
    train(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, criterion=criterion, save_dir=config["save_dir"], n_epochs=config["n_epochs"], save_freq=config["save_freq"], device=device)
    
    # test(test_loader, model, criterion)


    
if __name__ == "__main__":
    
    project_name = config["project_name"]

    wandb.init(
        project=project_name,
        config=config,
        mode="disabled"    
    )

    wandb.config.update(config)

    main()

    wandb.finish()