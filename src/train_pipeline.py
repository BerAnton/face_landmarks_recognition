import os
import yaml
import logging
import math
from pathlib import Path
from typing import Callable

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.optim import AdamW, Optimizer, lr_scheduler

from src.model import LandmarkModel
from src.data import ThousandLandmarksDataset
from src.transforms import ScaleMinSideToSize, CropCenter, TransformByKeys


def train(
    model: LandmarkModel, loader: DataLoader, loss_fn: Callable, optimizer: Optimizer, device: torch.device
) -> float:
    """Train loop for model.
    :args:
         - model - torch model to train.
         - loader - torch dataloader for train dataset.
         - loss_fn - train loss function.
         - device - torch.device ("cpu", "cuda")"""
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="Training..."):
        images = batch["image"].to(device)  # B * 3 * CROP_SIZE * CROP_SIZE
        landmarks = batch["landmarks"]  # B * (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)


def validate(model: LandmarkModel, loader: DataLoader, loss_fn: Callable, device: torch.device) -> float:
    """Function for model validation.
    :args:
         - model - torch model to validate.
         - loader - torch dataloader with validation dataset.
         - loss_fn - torch loss function.
         - device - torch.device ("cpu", "cuda")"""
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="Validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks)
        val_loss.append(loss.item())

    return np.mean(val_loss)


def train_pipeline(train_config_path: Path) -> None:
    """Train pipeline for landmarks recognition.
    :args:
         - train_config_path - path to config with train params."""

    with open(train_config_path, "r") as fin:
        train_config = yaml.safe_load(fin)

    input_data_path = train_config["input_data_path"]
    model_save_path = train_config["model_save_path"]
    model_name = train_config["model_name"]
    train_size = train_config["train_dataset_size"]
    batch_size = train_config["batch_size"]
    crop_size = train_config["crop_size"]
    lr = train_config["learning_rate"]
    epochs = train_config["epochs"]
    num_pts = train_config["num_pts"]
    cuda = train_config["use_cuda"]

    train_transforms = transforms.Compose(
        [
            ScaleMinSideToSize((crop_size, crop_size)),
            CropCenter(crop_size),
            TransformByKeys(transforms.ToPILImage(), ("image",)),
            TransformByKeys(transforms.ToTensor(), ("image",)),
            TransformByKeys(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
                ("image",),
            ),
        ]
    )

    print("Data loading")
    train_dataset = ThousandLandmarksDataset(input_data_path, train_transforms, split="train", train_size=train_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    val_dataset = ThousandLandmarksDataset(input_data_path, train_transforms, split="val", train_size=train_size)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    device = torch.device("cuda") if cuda else torch.device("cpu")

    print("Model creation")
    # define model
    model = LandmarkModel(num_pts)
    model.to(device)
    model.requires_grad_(True)

    # define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, amsgrad=True)
    Q = math.floor(len(train_dataset) / batch_size)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, Q)
    loss_fn = nn.L1Loss()

    # train loop
    best_val_loss = np.inf
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, loss_fn, optimizer, device=device)
        val_loss = validate(model, val_dataloader, loss_fn, device=device)
        scheduler.step(val_loss)
        print("Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}".format(epoch + 1, train_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(os.path.join(model_save_path, model_name), "wb") as fp:
                torch.save(model.state_dict(), fp)
