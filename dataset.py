import os
from skimage import io
import pandas as pd
from torch.utils.data import Dataset

class ACIDataset(Dataset):
    def __init__(self,
                 img_dir,
                 annotations_file=None,
                 transform=None,
                 target_transform=None):
        self.img_dir = img_dir
        self.is_labeled = False
        if annotations_file is not None:
            self.is_labeled = True
            self.img_labels = pd.read_csv(annotations_file)
        else:
            self.img_labels = pd.DataFrame(os.listdir(img_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,
                                self.img_labels.iloc[idx, 0])
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        if self.is_labeled:
            label = self.img_labels.iloc[idx, 1]
            if self.target_transform:
                label = self.target_transform(label)
            sample = [image, label]
        else:
            sample = [image, self.img_labels.iloc[idx, 0]]
        return sample
