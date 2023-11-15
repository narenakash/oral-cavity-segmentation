import torch

import time
import wandb
from tqdm import tqdm

from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose

from utils import load_checkpoint, save_checkpoint


def train(model, train_loader, val_loader, optimizer, criterion, save_dir, device, n_epochs=10, save_freq=1, n_gpus=4):

    best_val_dsc = -1
    best_val_dsc_epoch = -1
    
    for epoch in range(n_epochs):

        # train_epoch and eval_epoch functions
        train_loss, train_dsc = train_epoch(train_loader, model, optimizer, criterion, device)

        val_loss, val_dsc = eval_epoch(val_loader, model, criterion, device)

        if val_dsc > best_val_dsc:
            best_val_dsc = val_dsc
            best_val_dsc_epoch = epoch + 1

        #     save_checkpoint(model, optimizer, save_dir, epoch+1)
        #     print(f"New best DSC metric checkpoint saved at {save_dir}")

        print("Current epoch: {} current mean val dice: {:.4f} best mean val dice: {:.4f} at epoch {}".format(
                        epoch + 1, val_dsc, best_val_dsc, best_val_dsc_epoch))
        
        # save_checkpoint function
        # if epoch % save_freq == 0:
        #     save_checkpoint(model, optimizer, save_dir, epoch)
        #     print(f"Epoch {epoch} completed. Checkpoint saved at {save_dir}")

        # if n_gpus > 1:
        #     torch.save(model.module.state_dict(), f"{save_dir}/model_{epoch}.pth")
        # else:
        #     torch.save(model.state_dict(), f"{save_dir}/model_{epoch}.pth")


def train_epoch(loader, model, optimizer, criterion, device):
    model.train()

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    losses = []
    
    for batch_idx, (images, masks) in enumerate(tqdm(loader, total=len(loader))):        
        images = images.to(device)
        masks = masks.float().to(device)
        
        masks_pred = model(images)

        loss = criterion(masks_pred, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"{batch_idx}/{len(loader)}, train_loss: {loss.item():.4f}")
        losses.append(loss.item())

        masks_pred = post_transform(masks_pred)
        dice_metric(y_pred=masks_pred, y=masks)
    
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()

    loss = sum(losses) / len(losses)
    wandb.log({"train_loss": loss, "train_dice_score": dice_score})

    print(f"train_loss: {loss:.4f}, train_dice_score: {dice_score:.4f}")

    return loss, dice_score


def eval_epoch(loader, model, criterion, device):
    model.eval()

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    losses = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(loader, total=len(loader))):
            images = images.to(device)
            masks = masks.float().to(device)

            masks_pred = model(images)

            loss = criterion(masks_pred, masks)

            losses.append(loss.item())

            masks_pred = post_transform(masks_pred)
            dice_metric(y_pred=masks_pred, y=masks)

    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()

    loss = sum(losses) / len(losses)
    wandb.log({"val_loss": loss, "val_dice_score": dice_score})

    print(f"val_loss: {loss:.4f}, val_dice_score: {dice_score:.4f}")

    return loss, dice_score