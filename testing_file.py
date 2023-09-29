import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image

from natsort import natsorted
from plantdisease.data.helpers.helper_functions import find_classes, get_image_list, get_default_device
from plantdisease.data.datasets import CustomDataset
from plantdisease.data import DeviceDataLoader

train_dir = '/home/manojlovska/Documents/Projects/Plant-Disease-Detection/datasets/PlantVillage/Plant_leave_diseases_dataset_without_augmentation/train'
val_dir = '/home/manojlovska/Documents/Projects/Plant-Disease-Detection/datasets/PlantVillage/Plant_leave_diseases_dataset_without_augmentation/val'

train_data = get_image_list(train_dir)

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = get_image_list(train_dir)
val_data = get_image_list(val_dir)

train_data_custom = CustomDataset(target_dir=train_dir,
                           transform=train_transforms)

val_data_custom = CustomDataset(target_dir=val_dir,
                           transform=train_transforms)

# print(train_data.classes, train_data.class_to_idx)
print(len(train_data_custom), len(val_data_custom))

# batch size
batch_size = 16
train_dl = DataLoader(train_data_custom, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(val_data_custom, batch_size, num_workers=2, pin_memory=True)

print(train_dl)
print(val_dl)

device = "cpu"
print(device)
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

print(train_dl)
print(val_dl)


