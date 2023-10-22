import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from dataset import OPMDDataset

from utils import (
    load_checkpoint,
    save_checkpoint,
)


def train(loader, model, optimizer, criterion, scaler):
    
    for batch_idx, (data, targets) in enumerate(tqdm(loader)):
        data = data.to(device)
        targets = targets.float()

