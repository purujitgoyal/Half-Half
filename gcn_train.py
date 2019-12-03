import copy
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score

from glore.glore_model import GloreModel
from gcn.gcn_model import GcnModel
from Data.visual_genome import vg_data_load


def get_accuracy_score(y_true, y_pred):
    y_true_np = y_true.detach().numpy()
    y_pred_np = y_pred.detach().numpy()

    true_index = np.argmax(y_true_np, axis=0)
    y_pred_corr = y_pred_np[true_index]

    n_corr = np.sum(y_pred_corr == 1)
    return torch.tensor(n_corr)


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
            running_recall = 0.0

            # Iterate over data.
            for inputs, labels_ohe in dataloaders[phase]:
                inputs = inputs.to(device)
                labels_ohe = labels_ohe.to(device)
                # print(inputs.size())
                # print(labels_ohe.size())

                # zero the parameter gradients
                optimizer.zero_grad()

                embedding = torch.from_numpy(embedding_inp)
                inp_var = torch.autograd.Variable(embedding, requires_grad=False).float().detach()  # one hot. # pickle of word2vec.

                inp_var = inp_var.unsqueeze(0)  # add batch size dummy
                inp_var = inp_var.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, inp_var)
                    loss = criterion(outputs, labels_ohe.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                preds_ = torch.sigmoid(outputs)
                preds_np = preds_.cpu().detach().numpy()
                preds_np = np.where(preds_np > 0.5, 1, 0)
                preds_ = torch.from_numpy(preds_np)
                labels_ohe = labels_ohe.cpu()

                # if phase == 'train':
                #     running_corrects += accuracy_score(labels_ohe, preds_, normalize=False)
                # if phase == 'val':
                #     running_corrects += accuracy_score(labels_ohe, preds_, normalize=False)

                if phase == 'train':
                    running_recall += recall_score(labels_ohe, preds_, average="micro")
                if phase == 'val':
                    running_recall += recall_score(labels_ohe, preds_, average="micro")

                if phase == 'train':
                    running_corrects += get_accuracy_score(labels_ohe, preds_)
                if phase == 'val':
                    running_corrects += get_accuracy_score(labels_ohe, preds_)

                # print(multilabel_confusion_matrix(labels_ohe, preds_))
                # print(classification_report(labels_ohe, preds_))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_loss = running_loss
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_recall = running_recall / dataset_sizes[phase]
            # epoch_acc = running_corrects

            print('{} Loss: {:.4f} Acc: {:.4f} Recall: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_recall))

            # deep copy the model
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
    train_csv = "../data/visual_genome/sample/sample_visual_genome_train_ann.csv"
    val_csv = "../data/visual_genome/sample/sample_visual_genome_val_ann.csv"
    data_dir = "../data/visual_genome/sample/"
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

    gcn = GcnModel()
    finetune_conv = False
    model_dir = model_dir + str(finetune_conv)
    num_classes = 78
    # model = bm.get_model(finetune_conv=finetune_conv, device=device)
    model = gcn.get_model(t=0.4, adj_file=adj_file, out_features=num_classes, finetune_conv=finetune_conv, device=device)
    dataloaders, dataset_sizes = vg_data_load.get_data_loaders(train_csv=train_csv, val_csv=val_csv, data_dir=data_dir)

    cross_entropy = nn.BCEWithLogitsLoss()
    sgd = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.01)
    # adam = optim.Adam(model.parameters(), lr=0.9, weight_decay=0.01)

    exp_lr_scheduler = lr_scheduler.StepLR(sgd, step_size=40, gamma=0.9)

    with open(pkl_file, "rb") as f:
        inp = pickle.load(f)

    train_model(model, inp, criterion=cross_entropy, optimizer=sgd, scheduler=exp_lr_scheduler, dataloaders=dataloaders,
                dataset_sizes=dataset_sizes, device=device, model_dir=model_dir, num_epochs=num_epochs)
