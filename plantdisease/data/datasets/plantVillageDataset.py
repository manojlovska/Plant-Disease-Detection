import os
import pathlib
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
from typing import Tuple, Dict, List

from natsort import natsorted
from plantdisease.data.helpers.helper_functions import find_classes, get_image_list

class PlantDiseaseDataset(Dataset):
    def __init__(self, target_dir: str, transform=None) -> None:
        self.paths = get_image_list(target_dir)
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_dir)

    def load_image(self, index:int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path).convert('RGB')
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = os.path.dirname(self.paths[index]).split("/")[-1]
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
    









