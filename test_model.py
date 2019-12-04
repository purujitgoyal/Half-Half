import os
import sys

import torch
import torch.nn as nn

import data.data_load
from baseline import base_model
from dataset.half_half_dataset import HalfHalfLabelsDataset
from gcn import gcn_model
from glore import glore_model
import numpy as np
import matplotlib.pyplot as plt


def get_rank1_corrects(outputs, labels):
    rank1_acc = 0

    for i in range(outputs.size()[0]):
        output_i = outputs[i][labels[i]]
        _, max_index = torch.max(output_i, 0)
        if max_index == 0:
            rank1_acc += 1

    return torch.as_tensor(rank1_acc)


def get_MRR_corrects(outputs, labels):
    mrr_acc = 0

    for i in range(outputs.size()[0]):
        output_i = outputs[i][labels[i]]
        # first value in label[i] is the correct label
        score_correct = output_i[0]  # Score of correct label
        sorted_outputs, _ = torch.sort(output_i, descending=True)

        # Get rank of correct label in the list
        r_i = (sorted_outputs == score_correct).nonzero().cpu().numpy()[0][0] + 1
        mrr_acc += 1 / r_i

    return mrr_acc


def test_model(model, criterion, dataloaders, dataset_sizes, device):
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    running_mrr = 0

    # Iterate over data.
    for inputs, labels_ohe, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels_ohe = labels_ohe.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels_ohe.float())

        # statistics
        running_loss += loss.item() * inputs.size(0)
        _, target = labels_ohe.max(1)
        running_corrects += get_rank1_corrects(outputs, labels)
        running_mrr += get_MRR_corrects(outputs, labels)

        print(running_corrects)

    epoch_loss = running_loss / dataset_sizes['test']
    epoch_acc = running_corrects.double() / dataset_sizes['test']
    mrr = running_mrr / dataset_sizes['test']

    print('{} Loss: {:.4f} Rank-1 Acc: {:.4f} MRR: {:.4f}'.format(
        'val', epoch_loss, epoch_acc, mrr))
    print()


def test_model_image(model, idx, data_dir):
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    running_mrr = 0

    test_dataset = HalfHalfLabelsDataset(csv_file=test_csv, root_dir=os.path.join(data_dir, 'test', 'images'),
                                         transform=data.data_load.data_transformation(), num_classes=79)
    image, labels_ohe, labels = test_dataset.__getitem__(idx)
    imshow(image)

    labels_ohe = torch.from_numpy(labels_ohe).to(device)
    labels = labels.to(device)
    print(labels)
    # labels = torch.from_numpy(labels).to(device)
    image = torch.unsqueeze(image, 0)
    outputs = model(image)
    print(outputs.shape)
    _, preds = torch.max(outputs, 0)
    # loss = criterion(outputs, labels_ohe.float())

    # statistics
    # running_loss += loss.item() * inputs.size(0)
    print(labels_ohe)
    _, target = labels_ohe.max(0)
    running_corrects += get_rank1_corrects(outputs, labels)
    # running_mrr += get_MRR_corrects(outputs, labels)
    print(running_corrects)

    # epoch_loss = running_loss / dataset_sizes['test']
    # epoch_acc = running_corrects.double() / dataset_sizes['test']
    # mrr = running_mrr / dataset_sizes['test']
    #
    # print('{} Loss: {:.4f} Rank-1 Acc: {:.4f} MRR: {:.4f}'.format(
    #     'val', epoch_loss, epoch_acc, mrr))
    # print()

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
    cmd_args = sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_csv = './data/test_ann_encoded.csv'
    data_dir = '/Users/purujit/Desktop/'
    model_wts = './model_wts'

    bm = base_model.BaseModel()
    gm = glore_model.GloreModel()
    gcn = gcn_model.GcnModel()
    num_classes = 79
    finetune_conv = False

    model = bm.get_model(finetune_conv=finetune_conv, device=device)
    # model = gm.get_model(out_features=num_classes, finetune_conv=finetune_conv,
    #                      device=device)  # 79 classes for halfhalf dataset
    model.load_state_dict(torch.load(model_wts, map_location=torch.device('cpu')))

    # dataloaders, dataset_sizes = data.data_load.get_test_data_loader(test_csv=test_csv,
    #                                                                  data_dir=data_dir)

    cross_entropy = nn.BCEWithLogitsLoss()

    # test_model(model, criterion=cross_entropy, dataloaders=dataloaders,
    #            dataset_sizes=dataset_sizes, device=device)

    test_model_image(model, 50, data_dir)
