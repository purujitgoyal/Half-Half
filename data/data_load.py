import torch
from torchvision import models, transforms, datasets
import os
from torch.utils.data import DataLoader

from dataset.half_half_dataset import HalfHalfLabelsDataset
import matplotlib.pyplot as plt
import numpy as np
import nonechucks as nc


def data_transformation():
    transformation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    return transformation


def get_data_loaders(val_csv, data_dir, batch_size=32, num_workers=0, train_csv=None):
    transformation = _data_transformation()

    if train_csv is not None:
        # Training phase
        print('Loading Data in training mode')
        train_dataset = HalfHalfLabelsDataset(csv_file=train_csv, root_dir=os.path.join(data_dir, 'train', 'images'),
                                              transform=transformation, num_classes=79)
        val_dataset = HalfHalfLabelsDataset(csv_file=val_csv, root_dir=os.path.join(data_dir, 'val', 'images'),
                                            transform=transformation, num_classes=79)
        image_datasets = {'train': train_dataset, 'val': val_dataset}

        # print(image_datasets['train'][100])
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        return dataloaders, dataset_sizes

    else:
        # Validation phase
        print('Loading Data in validation mode')
        val_dataset = HalfHalfLabelsDataset(csv_file=val_csv, root_dir=os.path.join(data_dir, 'val', 'images'),
                                            transform=transformation, num_classes=79)
        image_datasets = {'val': val_dataset}

        dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                       for x in ['val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}

        return dataloaders, dataset_sizes


def get_test_data_loader(test_csv, data_dir, batch_size=32, num_workers=0):
    transformation = _data_transformation()
    print('Loading Data in test mode')
    test_dataset = HalfHalfLabelsDataset(csv_file=test_csv, root_dir=os.path.join(data_dir, 'test', 'images'),
                                         transform=transformation, num_classes=79)
    test_dataset = nc.SafeDataset(test_dataset)
    image_datasets = {'test': test_dataset}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                   for x in ['test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

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
    dataloaders, dataset_sizes = get_data_loaders(train_csv="./sample/sample_train_ann_encoded.csv",
                                                  val_csv="./sample/sample_train_ann_encoded.csv",
                                                  data_dir="./sample")

    for inputs, label_ohe, labels in dataloaders['train']:
        # print(i_batch, sample_batched)
        # print(inputs)
        # print(label_ohe)
        print(labels)
        scores = torch.randn(4, 79, dtype=torch.double)
        print(scores)
        for i in range(scores.size()[0]):
            print(scores[i][labels[i]])

    # sample_images, sample_labels = next(iter(dataloaders['train']))
    # print(sample_labels)
    # print(sample_images.size())
    # imshow(sample_images[0])
    # except:
    #     print("Exception")
