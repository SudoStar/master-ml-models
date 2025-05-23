import numpy as np
import torch
from . import transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import rasterio


def load_multiband(path):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)
    # return Image.open(path).convert("RGB")


def load_grayscale(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)
    # return Image.open(path).convert("L")


class OpenEarthMapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        msk_list,
        classes,
        img_size=512,
        augm=None,
        mu=None,
        sig=None,
    ):
        self.fn_msks = [str(f) for f in msk_list]
        self.fn_imgs = [f.replace("/labels/", "/images/") for f in self.fn_msks]
        self.size = img_size
        self.augm = augm
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = self.load_multiband(self.fn_imgs[idx])
        msk = self.load_grayscale(self.fn_msks[idx])

        data = self.to_tensor(self.augm({"image": img, "mask": msk}, self.size))
        return {"x": data["image"], "y": data["mask"], "fn": self.fn_msks[idx]}

    def __len__(self):
        return len(self.fn_imgs)


def to_tensor(sample, classes):
    masks = [(sample["mask"] == v) * 1 for v in classes]
    sample["mask"] = TF.to_tensor(np.stack(masks, axis=-1))
    sample["img"] = TF.to_tensor(sample["img"])
    return sample


class OpenEarthMapDatasetAlt(torch.utils.data.Dataset):
    def __init__(
        self,
        msk_list,
        num_classes,
        augm=None,
    ):
        self.fn_msks = [str(f) for f in msk_list]
        self.fn_imgs = [f.replace("/labels/", "/images/") for f in self.fn_msks]
        self.classes = np.arange(num_classes).tolist()
        self.augm = augm
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = Image.fromarray(self.load_multiband(self.fn_imgs[idx]))
        mask = Image.fromarray(self.load_grayscale(self.fn_msks[idx]))

        # data = self.to_tensor(self.augm({"image": img, "mask": msk}, self.size))
        img, mask = self.augm(img, mask)
        # img = torch.from_numpy(img).permute(2, 0, 1).float()
        # mask = torch.from_numpy(mask).long()
        data = to_tensor(
            sample={
                "img": np.array(img, dtype="uint8"),
                "mask": np.array(mask, dtype="uint8"),
            },
            classes=self.classes,
        )
        img_id = Path(self.fn_msks[idx]).name
        return {"img": data["img"], "gt_semantic_seg": data["mask"], "img_id": img_id}

    def __len__(self):
        return len(self.fn_imgs)
