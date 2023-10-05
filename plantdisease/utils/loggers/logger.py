import inspect
import os
import sys
from collections import defaultdict
from loguru import logger

import cv2
import numpy as np

import torch


class StreamToLoguruLogger:
    """
    stream object that redirects writes to a logger instance.
    """

    def __init__(self, level="INFO"):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        full_name = inspect.stack()[1].function
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        for line in buf.rstrip().splitlines():
            # use caller level log
            logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        # flush is related with CPR(cursor position report) in terminal
        return sys.__stdout__.flush()


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguruLogger(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def setup_logger(save_dir, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    logger.add(
        sys.stderr,
        format=loguru_format,
        level="INFO",
        enqueue=True,
    )
    logger.add(save_file)

    # redirect stdout/stderr to loguru
    redirect_sys_output("INFO")