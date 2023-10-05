import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from plantdisease.utils.plant_disease_class import PlantsClass

def make_parser():
    parser = argparse.ArgumentParser("Plant Disease Parser")

    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )

    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    return parser

@logger.catch
def main(conf_cls: PlantsClass, arguments):
    configuration_class = conf_cls()
    trainer = configuration_class.get_trainer(args=arguments)

    trainer.train()

if __name__ == "__main__":
    args_main = make_parser().parse_args()
    conf_cls = PlantsClass

    if not args_main.experiment_name:
        args_main.experiment_name = "Experiment"

    main(conf_cls, args_main)
