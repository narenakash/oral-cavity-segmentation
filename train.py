import torch

import time
from tqdm import tqdm

# from utils import (
#     load_checkpoint,
#     save_checkpoint,
# )


def train(model, train_loader, val_loader, optimizer, criterion, save_dir, device, n_epochs=10, save_freq=1):
    
    for epoch in range(n_epochs):
        train_epoch(train_loader, model, optimizer, criterion, device)

        eval_epoch(val_loader, model, criterion, device)
        
        # if epoch % save_freq == 0:
        #     save_checkpoint(model, optimizer, save_dir, epoch)
        #     print(f"Checkpoint saved at {save_dir}")

        # if n_gpus > 1:
        #     torch.save(model.module.state_dict(), f"{save_dir}/model_{epoch}.pth")
        # else:
        #     torch.save(model.state_dict(), f"{save_dir}/model_{epoch}.pth")

        # print(f"Epoch {epoch} completed. Checkpoint saved at {save_dir}")

        # val_loss = eval_epoch(val_loader, model, criterion, scaler)
        # print(f"Validation loss: {val_loss}")


def train_epoch(loader, model, optimizer, criterion, device):
    model.train()

    losses = []
    
    for batch_idx, (images, masks) in enumerate(tqdm(loader, total=len(loader))):        
        images = images.to(device)
        masks = masks.float().to(device)
        
        masks_pred = model(images)

        loss = criterion(masks_pred, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())


def eval_epoch(loader, model, criterion, device):
    model.eval()

    losses = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, total=len(loader))):
            images = images.to(device)
            masks = masks.float().to(device)

            masks_pred = model(images)

            loss = criterion(masks_pred, masks)

            losses.append(loss.item())