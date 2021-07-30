import os
import zipfile
from torch.utils.data import DataLoader, random_split
from dataset import ACIDataset
from parameters import *

if not os.path.exists(TRAIN_DIR):
    with zipfile.ZipFile(TRAIN_ZIP_DIR, 'r') as zip_ref:
        zip_ref.extractall('./')
if not os.path.exists(TEST_DIR):
    with zipfile.ZipFile(TEST_ZIP_DIR, 'r') as zip_ref:
        zip_ref.extractall('./')

data = ACIDataset(
    img_dir=TRAIN_DIR,
    annotations_file=ANNOTATIONS_DIR,
    transform=TRANSFORM,
    target_transform=None)

test_data = ACIDataset(
    img_dir=TEST_DIR,
    transform=TRANSFORM)

validation_size = int(len(data) * VALIDATION_PERC * 10 // 10)
auc_size = int(len(data) * AUC_PERC * 10 // 10)

train_data, val_data, auc_data = random_split(data, [len(data) - validation_size - auc_size,
                                                     validation_size,
                                                     auc_size])

train_dl = DataLoader(train_data, batch_size=BATCH_SIZE)
valid_dl = DataLoader(val_data, batch_size=BATCH_SIZE)
auc_dl   = DataLoader(auc_data, batch_size=1)
all_dl   = DataLoader(data, batch_size=BATCH_SIZE)
test_dl  = DataLoader(test_data, batch_size=BATCH_SIZE)
