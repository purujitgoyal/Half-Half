import torch
from torchvision import models, transforms, datasets
import os
from torch.utils.data import DataLoader

from dataset.half_half_dataset import HalfHalfLabelsDataset
import matplotlib.pyplot as plt
import numpy as np


def _data_transformation():
    transformation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    return transformation


def get_data_loaders(csv_file, data_dir, batch_size=32, num_workers=0):
    transformation = _data_transformation()
    # train_dataset = HalfHalfLabelsDataset
    image_datasets = {x: HalfHalfLabelsDataset(csv_file=csv_file, root_dir=os.path.join(data_dir, x),
                                               transform=transformation, num_classes=79)
                      for x in ['train', 'val']}

    # print(image_datasets['train'][100])
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)
    plt.show()


if __name__ == '__main__':
    dataloaders, dataset_sizes = get_data_loaders(csv_file="./sample/train_ann_encoded.csv", data_dir="./sample")

    for input, labels in dataloaders['train']:
        # print(i_batch, sample_batched)
        print(input)
        print(labels)

    # sample_images, sample_labels = next(iter(dataloaders['train']))
    # print(sample_labels)
    # print(sample_images.size())
    # imshow(sample_images[0])
    # except:
    #     print("Exception")
