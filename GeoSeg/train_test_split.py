import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random

OEM_ROOT = "./demo/"
OEM_DATA_DIR = "OpenEarthMap/"
TRAIN_TEST_LIST = OEM_DATA_DIR + "train.txt"
VAL_LIST = OEM_DATA_DIR + "val.txt"
TEST_LIST = OEM_DATA_DIR + "test.txt"


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    seed_everything(42)

    img_pths = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/labels/" in str(f)]

    train_test_pths = [
        str(f) for f in img_pths if f.name in np.loadtxt(TRAIN_TEST_LIST, dtype=str)
    ]

    random.shuffle(train_test_pths)

    training_pths = train_test_pths[:2500]
    testing_pths = train_test_pths[2500:]

    with open(OEM_DATA_DIR + "train_new.txt", "w") as f:
        for line in training_pths:
            filename = Path(line).name
            f.write(f"{filename}\n")

    with open(OEM_DATA_DIR + "test_new.txt", "w") as f:
        for line in testing_pths:
            filename = Path(line).name
            f.write(f"{filename}\n")


if __name__ == "__main__":
    main()
