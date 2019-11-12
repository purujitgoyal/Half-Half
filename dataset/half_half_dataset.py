"""Half Half dataset."""

import pandas as pd
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor


class HalfHalfLabelsDataset(Dataset):

    def __init__(self, csv_file, root_dir, num_classes, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels_df.iloc[idx, 0])
        image = Image.open(img_name)
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')
        image = ToTensor()(image)
        label = self.labels_df.iloc[idx, 1]
        label_ohe = np.zeros(self.num_classes,)
        label_ohe[label] = 1
        # print(type(label))
        # print(type(label_ohe))
        if self.transform:
            image = self.transform(image)

        # sample = {'image': image, 'label': label}

        return image, label_ohe.astype(int)
