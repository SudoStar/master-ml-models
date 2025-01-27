import glob
import os
import numpy as np
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import random

SEED = 42

CLASSES = (
    "bareland",
    "grass",
    "pavement",
    "road",
    "tree",
    "water",
    "cropland",
    "buildings",
)

PALETTE = [
    [128, 0, 0],
    [0, 255, 36],
    [148, 148, 148],
    [255, 255, 255],
    [34, 97, 38],
    [0, 69, 255],
    [75, 181, 73],
    [222, 31, 7],
]

class_rgb = {
    "Bareland": [128, 0, 0],
    "Grass": [0, 255, 36],
    "Pavement": [148, 148, 148],
    "Road": [255, 255, 255],
    "Tree": [34, 97, 38],
    "Water": [0, 69, 255],
    "Cropland": [75, 181, 73],
    "buildings": [222, 31, 7],
}

class_gray = {
    "Bareland": 1,
    "Grass": 2,
    "Pavement": 3,
    "Road": 4,
    "Tree": 5,
    "Water": 6,
    "Cropland": 7,
    "buildings": 8,
}


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", default="OpenEarthMap/**/labels")
    parser.add_argument("--output-mask-dir", default="outputs/labels")
    return parser.parse_args()


def convert_label(mask):
    mask[mask == 0] = 8
    mask -= 1

    return mask


def label2rgb(mask):
    """
    mask: labels (HxW)
    """
    out = np.zeros(shape=mask.shape + (3,), dtype="uint8")
    for k, v in class_gray.items():
        out[mask == v, 0] = class_rgb[k][0]
        out[mask == v, 1] = class_rgb[k][1]
        out[mask == v, 2] = class_rgb[k][2]
    return out


def patch_format(inp):
    (mask_path, masks_output_dir) = inp
    # print(mask_path, masks_output_dir)
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    label = convert_label(mask)
    rgb_label = label2rgb(label.copy())
    rgb_label = cv2.cvtColor(rgb_label, cv2.COLOR_RGB2BGR)
    out_mask_path_rgb = os.path.join(
        masks_output_dir + "_rgb", "{}.png".format(mask_filename)
    )
    cv2.imwrite(out_mask_path_rgb, rgb_label)

    out_mask_path = os.path.join(masks_output_dir, "{}.png".format(mask_filename))
    cv2.imwrite(out_mask_path, label)


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    masks_dir = args.mask_dir
    masks_output_dir = args.output_mask_dir
    mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))

    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
        os.makedirs(masks_output_dir + "_rgb")

    inp = [(mask_path, masks_output_dir) for mask_path in mask_paths]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print("images splitting spends: {} s".format(split_time))
