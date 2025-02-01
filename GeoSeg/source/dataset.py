import numpy as np
import torch
from . import transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path


def load_multiband(path):
    return Image.open(path).convert("RGB")


def load_grayscale(path):
    return Image.open(path).convert("L")


class OpenEarthMapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        msk_list,
        augm=None,
    ):
        self.fn_msks = [str(f) for f in msk_list]
        self.fn_imgs = [f.replace("/labels/", "/images/") for f in self.fn_msks]
        self.augm = augm
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = self.load_multiband(self.fn_imgs[idx])
        mask = self.load_grayscale(self.fn_msks[idx])

        img, mask = self.augm(img, mask)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = Path(self.fn_msks[idx]).name
        return {"img": img, "gt_semantic_seg": mask, "img_id": img_id}

    def __len__(self):
        return len(self.fn_imgs)
