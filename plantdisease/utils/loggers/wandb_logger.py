import inspect
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

import torch

class WandbLogger(object):
    def __init__(self,
                project=None,
                name=None,
                id=None,
                entity=None,
                save_dir=None,
                config=None,
                val_dataset=None,
                num_eval_images=100,
                log_checkpoints=False,
                **kwargs):
        """
        Args:
            project (str): wandb project name.
            name (str): wandb run name.
            id (str): wandb run id.
            entity (str): wandb entity name.
            save_dir (str): save directory.
            config (dict): config dict.
            val_dataset (Dataset): validation dataset.
            num_eval_images (int): number of images from the validation set to log.
            log_checkpoints (bool): log checkpoints
            **kwargs: other kwargs.
        """
        try:
            import wandb
            self.wandb = wandb
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "wandb is not installed."
                "Please install wandb using pip install wandb"
                )
        
        from plantdisease.data.datasets import PlantDiseaseDataset
