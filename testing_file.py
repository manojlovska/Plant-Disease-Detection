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
from plantdisease.data.datasets import PlantDiseaseDataset
from plantdisease.data import DeviceDataLoader

from plantdisease.models.ResNet9 import ResNet9
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
from torch.optim import SGD

train_dir = os.path.join(os.getcwd(), 'datasets/PlantVillage/Plant_leave_diseases_dataset_without_augmentation/train')
val_dir = os.path.join(os.getcwd(), 'datasets/PlantVillage/Plant_leave_diseases_dataset_without_augmentation/val')
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
classes, cls2idx = find_classes(train_dir)

# print(os.path.dirname(train_data[0]).split("/")[-1])
# print(classes)
# print(cls2idx)

val_data = get_image_list(val_dir)

train_data_custom = PlantDiseaseDataset(target_dir=train_dir,
                           transform=train_transforms)

val_data_custom = PlantDiseaseDataset(target_dir=val_dir,
                           transform=train_transforms)

# print(train_data.classes, train_data.class_to_idx)
print(len(train_data_custom), len(val_data_custom))

# batch size
batch_size = 8
train_dl = DataLoader(train_data_custom, batch_size, shuffle=True, num_workers=1, pin_memory=False)
val_dl = DataLoader(val_data_custom, batch_size, num_workers=1, pin_memory=False)

device = get_default_device()

print(torch.cuda.is_available())
print(device)
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

model = ResNet9(n_classes=39)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)

def training_step(model, batch):
    images, labels = batch
    out = model(images)                  # Generate predictions
    loss = F.cross_entropy(out, labels) # Calculate loss
    return loss

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# for epoch in range(1):
#     model.train()
#     train_losses = []
#     lrs = []

#     for batch in tqdm(train_dl):
#         loss = training_step(model, batch)
#         train_losses.append(loss)
#         loss.backward()

#         optimizer.step()
#         optimizer.zero_grad()

#         # recording and updating learning rates
#         lrs.append(get_lr(optimizer))
#         scheduler.step()


