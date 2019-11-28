import copy
import sys
import time

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import baseline.model
from Data import data_load


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
        r_i = (sorted_outputs == score_correct).nonzero().data.cpu().numpy()[0] + 1
        mrr_acc += 1/r_i

    return torch.as_tensor(mrr_acc)


def test_model(model, model_dir, criterion, dataloaders, dataset_sizes, device):
    since = time.time()
    model.load_state_dict(torch.load(model_dir, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    running_mrr = 0

    # Iterate over data.
    for inputs, labels_ohe, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels_ohe = labels_ohe.to(device)
        labels = labels.to(device)
        # print(inputs.size())
        # print(labels_ohe.size())

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels_ohe.float())

        # statistics
        running_loss += loss.item() * inputs.size(0)
        _, target = labels_ohe.max(1)

        running_corrects += get_rank1_corrects(outputs, labels)
        running_mrr += get_MRR_corrects(outputs, labels)

    loss = running_loss / dataset_sizes['val']
    acc = running_corrects.double() / dataset_sizes['val']
    mrr = running_mrr.double() / dataset_sizes['val']

    print('{} Loss: {:.4f} Rank-1 Acc: {:.4f} MRR: {:.4f}'.format(
        'val', loss, acc, mrr))
    print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    cmd_args = sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_csv = "./data/sample/sample_train_ann_encoded.csv"
    data_dir = "./data/sample"
    num_epochs = 25

    if len(cmd_args) != 4:
        print("Check your arguments")
        print("Running on sample data")
    else:
        val_csv = cmd_args[1]
        data_dir = cmd_args[2]
        model_dir = cmd_args[3]

    finetune_conv = False
    model = baseline.model.get_model(out_features=79, finetune_conv=finetune_conv, device=device)
    dataloaders, dataset_sizes = data_load.get_data_loaders(val_csv=val_csv, data_dir=data_dir)

    cross_entropy = nn.BCEWithLogitsLoss()

    test_model(model, model_dir=model_dir, criterion=cross_entropy, dataloaders=dataloaders,
               dataset_sizes=dataset_sizes, device=device)
