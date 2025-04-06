import argparse
from pathlib import Path
import glob
import ttach as tta
import cv2
import numpy as np
import torch
import albumentations as albu
from tools.cfg import py2cfg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_supervision import *
import random
import os


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def label2rgb(mask):
    """
    mask: labels (HxW)
    """
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 128]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [36, 255, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [148, 148, 148]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [38, 97, 34]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 69, 0]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [73, 181, 75]
    mask_rgb[np.all(mask_convert == 8, axis=0)] = [7, 31, 222]
    return mask_rgb


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "-i",
        "--image_path",
        type=Path,
        required=True,
        help="Path to  huge image folder",
    )
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg(
        "-o",
        "--output_path",
        type=Path,
        help="Path to save resulting masks.",
        required=True,
    )
    arg(
        "-t",
        "--tta",
        help="Test time augmentation.",
        default=None,
        choices=[None, "d4", "lr"],
    )
    arg("-ph", "--patch-height", help="height of patch size", type=int, default=512)
    arg("-pw", "--patch-width", help="width of patch size", type=int, default=512)
    arg("-b", "--batch-size", help="batch size", type=int, default=2)
    arg("-s", "--stride", help="stride for sliding window", type=int, default=256)
    return parser.parse_args()


def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]

    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
    # print(oh, ow, rh, rw, height_pad, width_pad)
    h, w = oh + height_pad, ow + width_pad

    pad = albu.PadIfNeeded(
        min_height=h,
        min_width=w,
        position="top_left",
        border_mode=0,
        value=[0, 0, 0],
    )(image=image)
    img_pad = pad["image"]
    return img_pad, height_pad, width_pad


class InferenceDataset(Dataset):
    def __init__(self, tile_list=None, positions=None, transform=albu.Normalize()):
        self.tile_list = tile_list
        self.positions = positions
        self.transform = transform

    def __getitem__(self, index):
        img = self.tile_list[index]
        x, y = self.positions[index]
        img_id = index
        aug = self.transform(image=img)
        img = aug["image"]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        results = dict(img_id=img_id, img=img, x=x, y=y)
        return results

    def __len__(self):
        return len(self.tile_list)


def make_dataset_for_one_huge_image(img_path, patch_size, stride):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tile_list = []
    positions = []
    image_pad, height_pad, width_pad = get_img_padded(img.copy(), patch_size)

    output_height, output_width = image_pad.shape[0], image_pad.shape[1]

    for x in range(0, output_height - patch_size[0] + 1, stride):
        for y in range(0, output_width - patch_size[1] + 1, stride):
            image_tile = image_pad[x : x + patch_size[0], y : y + patch_size[1]]
            tile_list.append(image_tile)
            positions.append((x, y))

    dataset = InferenceDataset(tile_list=tile_list, positions=positions)
    return (
        dataset,
        width_pad,
        height_pad,
        output_width,
        output_height,
        image_pad,
        img.shape,
    )


def main():
    args = get_args()
    seed_everything(42)
    patch_size = (args.patch_height, args.patch_width)
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
    )

    model.cuda()
    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1, 1.25, 1.5, 1.75]),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    img_paths = []
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    for ext in ("*.tif", "*.png", "*.jpg"):
        img_paths.extend(glob.glob(os.path.join(args.image_path, ext)))
    img_paths.sort()
    # print(img_paths)

    stride = args.stride

    for img_path in img_paths:
        img_name = img_path.split("/")[-1]
        # print('origin mask', original_mask.shape)
        (
            dataset,
            width_pad,
            height_pad,
            output_width,
            output_height,
            img_pad,
            img_shape,
        ) = make_dataset_for_one_huge_image(img_path, patch_size, stride)
        # print('img_padded', img_pad.shape)
        output_mask = np.zeros(shape=(output_height, output_width), dtype=np.uint8)

        num_classes = config.num_classes  # Ensure this is accessible
        accumulator = np.zeros(
            (output_height, output_width, num_classes), dtype=np.float32
        )
        count_accumulator = np.zeros((output_height, output_width), dtype=np.uint32)

        with torch.no_grad():
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                drop_last=False,
                shuffle=False,
            )
            for input in tqdm(dataloader):
                raw_predictions = model(input["img"].cuda())
                probabilities = torch.nn.Softmax(dim=1)(raw_predictions).cpu().numpy()
                xs = input["x"].numpy()
                ys = input["y"].numpy()

                for i in range(probabilities.shape[0]):
                    x_start = xs[i]
                    y_start = ys[i]
                    prob = probabilities[i].transpose(1, 2, 0)  # HWC

                    # Update accumulators
                    accumulator[
                        x_start : x_start + patch_size[0],
                        y_start : y_start + patch_size[1],
                    ] += prob
                    count_accumulator[
                        x_start : x_start + patch_size[0],
                        y_start : y_start + patch_size[1],
                    ] += 1

        average_probs = accumulator / (count_accumulator[..., np.newaxis] + 1e-8)
        output_mask = np.argmax(average_probs, axis=2).astype(np.uint8)
        output_mask = output_mask[: img_shape[0], : img_shape[1]]  # Remove padding

        output_mask = label2rgb(output_mask)
        cv2.imwrite(os.path.join(args.output_path, img_name), output_mask)


if __name__ == "__main__":
    main()
