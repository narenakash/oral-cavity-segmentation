import torch

import time
import wandb
from tqdm import tqdm

from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose

from utils import load_checkpoint, save_checkpoint


def test(loader, model, criterion, device):
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
    wandb.log({"test_loss": loss, "test_dice_score": dice_score})

    print(f"test_loss: {loss:.4f}, test_dice_score: {dice_score:.4f}")

    return loss, dice_metric