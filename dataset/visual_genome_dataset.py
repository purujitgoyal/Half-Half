"""Visual Genome dataset."""
import json

import pandas as pd
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor


class VisualGenomeDataset(Dataset):

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
                                str(self.labels_df.iloc[idx, 0])+'.jpg')
        image = Image.open(img_name)
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')
        image = ToTensor()(image)
        labels = json.loads(self.labels_df.iloc[idx, 2])
        # print(type(labels))
        labels = torch.tensor(labels)
        label_ohe = np.zeros(self.num_classes,)
        label_ohe[labels] = 1

        if self.transform:
            image = self.transform(image)

        return image, label_ohe


if __name__ == '__main__':
    csv = pd.read_csv('../data/visual_genome/sample/sample_visual_genome_train_ann.csv')
    print(json.loads(csv.iloc[0, 2]))
