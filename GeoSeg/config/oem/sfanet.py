from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.models.SFANet import SFANet
from tools.utils import Lookahead
from tools.utils import process_model_params
from source.dataset import OpenEarthMapDataset
from pathlib import Path
import albumentations as albu
from geoseg.datasets.transform import *

CLASSES = (
    "unknown",
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
ignore_index = 0
train_batch_size = 2
val_batch_size = 2
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES
img_size = 512

weights_name = "sfanet-swin-512crop-ms-epoch45-rep"
weights_path = "model_weights/sfanet/{}".format(weights_name)
test_weights_name = weights_name
log_name = "sfanet/{}".format(weights_name)
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
TRAIN_LIST = OEM_DATA_DIR + "train_new.txt"
VAL_LIST = OEM_DATA_DIR + "val.txt"
TEST_LIST = OEM_DATA_DIR + "test_new.txt"
WEIGHT_DIR = OEM_ROOT + "weight"  # path to save weights
OUT_DIR = OEM_ROOT + "result/"  # path to save prediction images

#  define the network
net = SFANet(num_classes=num_classes)

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

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
