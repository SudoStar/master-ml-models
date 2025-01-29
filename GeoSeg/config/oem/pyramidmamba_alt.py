from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.models.PyramidMamba import PyramidMamba
from tools.utils import Lookahead
from tools.utils import process_model_params
from source.dataset import OpenEarthMapDatasetAlt
import random
from pathlib import Path
import albumentations as albu
import cv2
from geoseg.datasets.transform import *

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

# training hparam
max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 2
val_batch_size = 2
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES
img_size = 512

weights_name = "pyramidmamba-r18-512crop-ms-epoch45-rep"
weights_path = "model_weights/pyramidmamba/{}".format(weights_name)
test_weights_name = "last"
log_name = "pyramidmamba/{}".format(weights_name)
monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None  # the path for the pretrained model weight
gpus = "auto"  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

OEM_ROOT = "./demo/"
OEM_DATA_DIR = "OpenEarthMap/"
TRAIN_TEST_LIST = OEM_DATA_DIR + "train.txt"
VAL_LIST = OEM_DATA_DIR + "val.txt"
TEST_LIST = OEM_DATA_DIR + "test.txt"
WEIGHT_DIR = OEM_ROOT + "weight"  # path to save weights
OUT_DIR = OEM_ROOT + "result/"  # path to save prediction images

#  define the network
net = PyramidMamba(num_classes=num_classes)

# define the loss
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False

# define the dataloader


def get_training_transform():
    train_transform = [
        albu.Resize(height=img_size, width=img_size),
        albu.HorizontalFlip(p=0.5),
        albu.Normalize(),
    ]
    return albu.Compose(train_transform)


def get_val_transform():
    val_transform = [
        albu.Resize(height=img_size, width=img_size),
        albu.Normalize(),
    ]
    return albu.Compose(val_transform)


def train_aug(img, mask):
    crop_aug = Compose(
        [
            RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode="value"),
            SmartCropV1(
                crop_size=img_size,
                max_ratio=0.75,
                ignore_index=ignore_index,
                nopad=False,
            ),
        ]
    )
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug["image"], aug["mask"]
    return img, mask


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug["image"], aug["mask"]
    return img, mask


img_pths = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/labels/" in str(f)]

train_test_pths = [
    str(f) for f in img_pths if f.name in np.loadtxt(TRAIN_TEST_LIST, dtype=str)
]
val_pths = [str(f) for f in img_pths if f.name in np.loadtxt(VAL_LIST, dtype=str)]

print("Total samples      :", len(img_pths))
print("Training samples   :", len(train_test_pths))
print("Validation samples :", len(val_pths))

random.shuffle(train_test_pths)

training_pths = train_test_pths[:2500]
testing_pths = train_test_pths[2500:]

train_set = OpenEarthMapDatasetAlt(
    msk_list=training_pths,
    augm=train_aug,
)
valid_set = OpenEarthMapDatasetAlt(
    msk_list=val_pths,
    augm=val_aug,
)

train_loader = DataLoader(
    dataset=train_set,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=valid_set,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

# define the optimizer
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epoch, eta_min=1e-6
)
