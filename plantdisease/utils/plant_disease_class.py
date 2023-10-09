import ast
import pprint
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple
from tabulate import tabulate
from.base_class import BaseClass

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LinearLR
import os
from torchvision import transforms

class PlantsClass(BaseClass):
    """Basic class for any experiment."""
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        # number of classes
        self.num_classes = 39

        # -------------- dataloader config ------------- #
        # (height, width)
        self.input_size = (256, 256)

        # number of workers
        self.data_num_workers = 1

        # pin memory
        self.pin_memory = False

        # directory of dataset images - ENCODE YOUR OWN PATH
        self.data_dir = "/home/manojlovska/Documents/Projects/Plant-Disease-Detection/datasets/PlantVillage/Plant_leave_diseases_dataset_without_augmentation"

        # -------------- transform config -------------- #
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])


        # -------------- training config --------------- #
        # max training epochs
        self.max_epoch = 10
        self.max_lr = 0.01
        self.grad_clip = 0.1
        self.weight_decay = 1e-4

        # shuffle set to true for training
        self.shuffle_train = True

        # optimizer
        self.opt = torch.optim.Adam
        
        # learning rate scheduler, default is linear
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR

        # if set to 1, user could see log every iteration
        self.print_interval = 10

        # eval period in epoch
        self.eval_interval = 1

        # save history checkpoint or not
        self.save_history_ckpt = True

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (256, 256)

        # shuffle set to true for training
        self.shuffle_test = False
        self.return_predictions = True


    def get_model(self):
        from plantdisease.models.ResNet9 import ResNet9

        model = ResNet9(n_classes = self.num_classes)

        return model

    def get_dataset(self):
        from plantdisease.data.datasets import PlantDiseaseDataset

        return PlantDiseaseDataset(
            target_dir=os.path.join(self.data_dir,"train"),
            transform = self.train_transforms,
        )

    def get_data_loader(self, batch_size, device):
        from torch.utils.data import DataLoader
        from plantdisease.data.dataloader import DeviceDataLoader
        
        dl = DataLoader(self.get_dataset(), batch_size, shuffle=self.shuffle_train, num_workers=self.data_num_workers, pin_memory=self.pin_memory)

        return DeviceDataLoader(dl, device)
        
    
    def get_eval_dataset(self):
        from plantdisease.data.datasets import PlantDiseaseDataset

        return PlantDiseaseDataset(
            target_dir=os.path.join(self.data_dir, "val"),
            transform = self.val_transforms,
        )
    
    def get_eval_data_loader(self, batch_size, device):
        from torch.utils.data import DataLoader
        from plantdisease.data.dataloader import DeviceDataLoader
        
        dl = DataLoader(self.get_eval_dataset(), batch_size, shuffle=self.shuffle_test, num_workers=self.data_num_workers, pin_memory=self.pin_memory)

        return DeviceDataLoader(dl, device)


    def get_trainer(self, args):
        from plantdisease.utils.trainers.trainer import Trainer

        trainer = Trainer(self, args)

        return trainer

    
    def eval(self, model, val_loader, device):
        from plantdisease.utils.evaluators.evaluate import evaluate

        return evaluate(model, val_loader, device, return_predictions=self.return_predictions)