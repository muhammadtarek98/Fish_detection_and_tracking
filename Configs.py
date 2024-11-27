import albumentations as A
import cv2
import numpy as np
import torch
import torchvision
from Dataset import *
from albumentations.pytorch import ToTensorV2
import lightning
import pytorch_lightning as pl
from torch_model import *
def get_transforms(train: bool) -> A.Compose:
    transforms = []
    if train:
        transforms.append(A.ToFloat())
        transforms.append(A.Normalize(mean=[0,0,0], std=[1,1,1]))
        transforms.append(A.pytorch.ToTensorV2())
        return A.Compose(transforms=transforms,
                     bbox_params=A.BboxParams(format='pascal_voc',
                                              label_fields=['labels']))
    else:
        transforms.append(A.ToFloat())
        transforms.append(A.Normalize(mean=[0,0,0], std=[1,1,1]))
        transforms.append(A.pytorch.ToTensorV2())
        return A.Compose(transforms=transforms)
full_dataset = CustomDataset(
    image_dir="/home/muhammad/projects/SEE_assessment/DeepFish/Segmentation/images/valid",
    mask_dir="/home/muhammad/projects/SEE_assessment/DeepFish/Segmentation/masks/valid",
    transform=get_transforms(train=True),
)
num_devices = torch.cuda.device_count()
device_type ="gpu"if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoches=10
lr=1e-4
num_batches=1
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=num_batches,
                                         shuffle=True)
val_dataloader=torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=num_batches,
                                           shuffle=False)

logger=lightning.pytorch.loggers.TensorBoardLogger("logs/",name="FRCNN")
checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1,monitor="val_map",mode="max",save_on_train_epoch_end=True)
early_stop=pl.callbacks.EarlyStopping(monitor='train_total_loss', patience=10,min_delta=0.0)