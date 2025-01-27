from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.models.PyramidMamba import PyramidMamba
from tools.utils import Lookahead
from tools.utils import process_model_params
import source
import random

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
max_epoch = 30
ignore_index = len(CLASSES)
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "pyramid_mamba-r8-512crop-ms-epoch30-rep"
weights_path = "model_weights/oem/{}".format(weights_name)
test_weights_name = "last"
log_name = "oem/{}".format(weights_name)
monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None  # the path for the pretrained model weight
gpus = "auto"  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = PyramidMamba(num_classes=num_classes)

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True


def get_training_transform():
    train_transform = [albu.HorizontalFlip(p=0.5), albu.Normalize()]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose(
        [
            RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode="value"),
            SmartCropV1(
                crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False
            ),
        ]
    )
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug["image"], aug["mask"]
    return img, mask


OEM_DATA_DIR = "OpenEarthMap/"
TRAIN_TEST_LIST = OEM_DATA_DIR + "train.txt"
VAL_LIST = OEM_DATA_DIR + "val.txt"
batch_size = 4

img_pths = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/labels/" in str(f)]

train_test_pths = [
    str(f) for f in img_pths if f.name in np.loadtxt(TRAIN_TEST_LIST, dtype=str)
]
val_pths = [str(f) for f in img_pths if f.name in np.loadtxt(VAL_LIST, dtype=str)]

random.shuffle(train_test_pths)

training_pths = train_test_pths[:2500]
testing_pths = train_test_pths[2500:]

trainset = source.dataset.Dataset(training_pths, classes=classes, size=1024, train=True)
validset = source.dataset.Dataset(val_pths, classes=classes, train=False)
testset = source.dataset.Dataset(testing_pths, classes=classes, train=False)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

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
