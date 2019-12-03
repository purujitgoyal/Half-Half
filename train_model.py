import copy
import pickle
import sys
import time

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from baseline import base_model
from glore import glore_model
from gcn import gcn_model

import data.data_load


def get_rank1_corrects(outputs, labels):
    rank1_acc = 0

    for i in range(outputs.size()[0]):
        output_i = outputs[i][labels[i]]
        # print(output_i)
        _, max_index = torch.max(output_i, 0)
        if max_index == 0:
            rank1_acc += 1

    return torch.as_tensor(rank1_acc)


def train_model(model, embedding_inp, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, model_dir, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels_ohe, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels_ohe = labels_ohe.to(device)
                labels = labels.to(device)
                # print(inputs.size())
                # print(labels_ohe.size())

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                embedding = torch.from_numpy(embedding_inp)
                inp_var = torch.autograd.Variable(embedding).float().detach()  # one hot. # pickle of word2vec.

                inp_var = inp_var.unsqueeze(0)  # add batch size dummy
                inp_var = inp_var.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    # inputs: (batch x channels x 448 x 448)
                    # inp_var: (batch x num_classes x 300)
                    outputs = model(inputs, inp_var)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels_ohe.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                _, target = labels_ohe.max(1)
                if phase == 'train':
                    running_corrects += torch.sum(preds == target)
                if phase == 'val':
                    running_corrects += get_rank1_corrects(outputs, labels)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model

            # Todo: Apply early stopping using val_acc, plot losses

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_dir)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    cmd_args = sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_csv = "./data/sample/sample_train_ann_encoded.csv"
    val_csv = "./data/sample/sample_train_ann_encoded.csv"
    data_dir = "./data/sample"
    num_epochs = 25

    if len(cmd_args) != 8:
        print("Check your arguments")
        print("Running on sample data")
    else:
        train_csv = cmd_args[1]
        val_csv = cmd_args[2]
        data_dir = cmd_args[3]
        num_epochs = int(cmd_args[4])
        model_dir = cmd_args[5]
        pkl_file = cmd_args[6]
        adj_file = cmd_args[7]

    bm = base_model.BaseModel()
    gm = glore_model.GloreModel()
    gcn = gcn_model.GcnModel()
    num_classes = 79
    finetune_conv = False
    model_dir = model_dir + "_" + str(finetune_conv)
    # model = bm.get_model(finetune_conv=finetune_conv, device=device)
    # model = gm.get_model(out_features=num_classes, finetune_conv=finetune_conv, device=device)  # 79 classes for halfhalf dataset

    # Todo: add no_grad = True for word embeddings. Need to tune in training phase? check L421 engine.py (ML-GCN)

    model = gcn.get_model(t=0.4, adj_file=adj_file, out_features=num_classes, finetune_conv=finetune_conv, device=device)


    dataloaders, dataset_sizes = data.data_load.get_data_loaders(train_csv=train_csv, val_csv=val_csv, data_dir=data_dir)

    cross_entropy = nn.BCEWithLogitsLoss()
    sgd = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # adam = optim.Adam(model.parameters(), lr=0.9, weight_decay=0.01)

    exp_lr_scheduler = lr_scheduler.StepLR(sgd, step_size=30, gamma=0.1)

    with open(pkl_file, "rb") as f:
        inp = pickle.load(f)

    train_model(model, inp, criterion=cross_entropy, optimizer=sgd, scheduler=exp_lr_scheduler, dataloaders=dataloaders,
                dataset_sizes=dataset_sizes, device=device, model_dir=model_dir, num_epochs=num_epochs)
