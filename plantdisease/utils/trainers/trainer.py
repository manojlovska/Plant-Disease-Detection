import datetime
import os
import sys
import time
from loguru import logger

import torch

from plantdisease.utils.plant_disease_class import PlantsClass
from plantdisease.utils.loggers.logger import setup_logger
from plantdisease.data.helpers.helper_functions import get_default_device
import torch.nn.functional as F
from tqdm import tqdm
from dvclive import Live


def training_step(model, batch, device):
    images, labels = batch
    images.to(device)
    labels.to(device)
    out = model(images)
    out.to(device)                  # Generate predictions
    loss = F.cross_entropy(out, labels) # Calculate loss
    return loss, out

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer:
    def __init__(self, base_cls: PlantsClass, args):
        self.base_cls = base_cls
        self.args = args

        self.max_epoch = base_cls.max_epoch
        self.save_history_ckpt = base_cls.save_history_ckpt

        self.input_size = base_cls.input_size
        self.file_name = os.path.join(base_cls.output_dir, args.experiment_name)

        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.device = get_default_device()
        
        # Initialization of accuracies and losses
        self.best_acc = 0
        self.epoch_acc = 0
        self.val_loss = 0
        self.return_predictions = base_cls.return_predictions

        setup_logger(
            self.file_name,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def before_train(self):
        logger.info("args: {}".format(self.args)) 
        logger.info("Basic class value:\n{}".format(self.base_cls))

        # model related init
        model = self.base_cls.get_model()
        input_size = self.base_cls.input_size
        model.to(self.device)

        from torchsummary import summary
        logger.info(
            "Model Summary: {}".format(summary(model, (3, 224, 224)))
        )
        
        # solver related init
        self.optimizer = self.base_cls.opt

        # data related init
        self.train_loader = self.base_cls.get_data_loader(
            batch_size=self.args.batch_size,
            device=self.device
        )

        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.base_cls.scheduler

        self.model = model

        logger.info("Starting the training process ...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the highest precision is {:.2f}".format(self.best_acc * 100)
        )

    def train_in_epoch(self):
        with Live() as live:
            # live.log_param("device", self.device)
            live.log_param("batch size", self.args.batch_size)
            live.log_param("epochs", self.max_epoch)
            # live.log_param("optimizer", self.optimizer)

            for self.epoch in range(self.max_epoch):
                # Before epoch
                logger.info(" *** Start training epoch {} ***".format(self.epoch + 1))

                self.model.train()
                train_losses = []
                lrs = []

                iter_step = 0
                for batch in self.train_loader:
                    iter_step += 1
                    iter_start_time = time.time()
                    
                    loss, outputs = training_step(self.model, batch, self.device)
                    train_losses.append(loss)
                    loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # recording and updating learning rates
                    lr = get_lr(self.optimizer)
                    lrs.append(lr)
                    self.lr_scheduler.step()

                    iter_end_time = time.time()

                    live.log_metric("loss", loss.item())
                    live.log_metric("learning_rate", lr)
                    live.next_step()
                    
                    iter_time = iter_end_time - iter_start_time

                    logger.info("Epoch: {}/{}, iter: {}/{}, loss: {}, learning rate: {}, training time: {} s".format(self.epoch+1, self.max_epoch, iter_step, self.max_iter, loss, lr, iter_time))

                # After epoch
                self.save_ckpt(ckpt_name="latest")
                if (self.epoch + 1) % self.base_cls.eval_interval == 0:
                    self.evaluate_and_save_model()
                    live.log_metric("epoch_accuracy", self.epoch_acc.item())
                    live.log_metric("best_accuracy", self.best_acc.item())
                    live.log_metric("val_loss", self.val_loss.item())
    
    def evaluate_and_save_model(self):
        evalmodel = self.model
        self.val_loader = self.base_cls.get_eval_data_loader(
            batch_size=self.args.batch_size,
            device=self.device
        )
        
        output_dict = self.base_cls.eval(evalmodel, self.val_loader, self.device)
        
        self.epoch_acc = output_dict["val_accuracy"]
        self.val_loss = output_dict["val_loss"]

        update_best_ckpt = self.epoch_acc > self.best_acc
        self.best_acc = max(self.best_acc, self.epoch_acc)

        logger.info("Epoch accuracy: {}".format(self.epoch_acc))
        logger.info("Best accuracy: {}".format(self.best_acc))
        self.save_ckpt("last_epoch", update_best_ckpt, acc=self.epoch_acc)

        logger.info(" *** Training of epoch {} ended! Epoch accuracy: {}, best training accuracy achieved: {} ***".format(self.epoch + 1, self.epoch_acc, self.best_acc))

        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", acc=self.epoch_acc)

        
    def save_ckpt(self, ckpt_name, update_best_ckpt=False, acc=None):

        save_model = self.model
        logger.info("Saving weights to {}".format(self.file_name))
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_ap": self.best_acc,
            "curr_ap": acc,
        }
        filename = os.path.join(self.file_name, ckpt_name + "_ckpt.pth")
        torch.save(ckpt_state, filename)

        if update_best_ckpt:
            filename = os.path.join(self.file_name, "best_ckpt.pth")
            torch.save(ckpt_state, filename)



    




        
