from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.models.UNetFormer_lsk import UNetFormer_lsk_s
from pathlib import Path
import source
import random
import os


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

OEM_ROOT = "./demo/"
OEM_DATA_DIR = "OpenEarthMap/"
TRAIN_TEST_LIST = OEM_DATA_DIR + "train.txt"
VAL_LIST = OEM_DATA_DIR + "val.txt"
TEST_LIST = OEM_DATA_DIR + "test.txt"
WEIGHT_DIR = OEM_ROOT + "weight"  # path to save weights
OUT_DIR = OEM_ROOT + "result/"  # path to save prediction images
os.makedirs(WEIGHT_DIR, exist_ok=True)

seed = 0
batch_size = 1
n_epochs = 15
classes = [1, 2, 3, 4, 5, 6, 7, 8]
n_classes = len(classes) + 1
classes_wt = np.ones([n_classes], dtype=np.float32)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = "cuda" if torch.cuda.is_available() else "cpu"

img_size = 1024
name = "unet-former-lsknet"

#  define the network
net = UNetFormer_lsk_s(num_classes=n_classes)

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

trainset = source.dataset.OpenEarthMapDataset(
    msk_list=training_pths,
    classes=classes,
    img_size=img_size,
    augm=source.transforms.train_augm,
)
validset = source.dataset.OpenEarthMapDataset(
    msk_list=val_pths,
    classes=classes,
    img_size=img_size,
    augm=source.transforms.valid_augm,
)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=batch_size, shuffle=False)
