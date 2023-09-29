from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from natsort import natsorted 
import shutil

# Data directory
data_dir_src = '../datasets/PlantVillage/Plant_leave_diseases_dataset_without_augmentation/'
data_dir_dst = os.path.join(data_dir_src, 'all_data')

folders = natsorted(os.listdir(data_dir_src))

if not os.path.exists(data_dir_dst):
    os.mkdir(data_dir_dst)

for folder in folders:
    source = os.path.join(data_dir_src, folder)
    destination = os.path.join(data_dir_dst, folder)

    if not os.path.exists(destination):
        os.mkdir(destination)

    for image in natsorted(os.listdir(source)):
        shutil.move(os.path.join(source, image), destination)

    os.rmdir(source)

data_dir = data_dir_dst

train_dir = '../datasets/PlantVillage/Plant_leave_diseases_dataset_without_augmentation/train'
val_dir = '../datasets/PlantVillage/Plant_leave_diseases_dataset_without_augmentation/val'

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(val_dir):
    os.mkdir(val_dir)

for folder in natsorted(os.listdir(data_dir)):
    source = os.path.join(data_dir,folder)
    destination_train = os.path.join(train_dir, folder)
    destination_val = os.path.join(val_dir, folder)

    if not os.path.exists(destination_train):
        os.mkdir(destination_train)

    if not os.path.exists(destination_val):
        os.mkdir(destination_val)

    for image in natsorted(os.listdir(source))[:round(0.8*len(natsorted(os.listdir(source))))]:
        shutil.copy2(os.path.join(source,image), destination_train)
    for image in natsorted(os.listdir(source))[round(0.8*len(natsorted(os.listdir(source)))):]:
        shutil.copy2(os.path.join(source,image), destination_val)


    



