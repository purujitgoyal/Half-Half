import copy
import time

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import baseline.model
import data.data_load


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, device='cpu'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                print(preds)
                # print(labels.data)
                max_val, target = labels.max(1)
                print(target)
                running_corrects += torch.sum(preds == target)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    csv_file_path = "./data/sample/train_ann_encoded.csv"
    data_dir = "./data/sample"
    finetune_conv = False
    model = baseline.model.get_model(out_features=79, finetune_conv=finetune_conv, device=device)
    dataloaders, dataset_sizes = data.data_load.get_data_loaders(csv_file=csv_file_path, data_dir=data_dir)

    cross_entropy = nn.BCEWithLogitsLoss()
    sgd = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(sgd, step_size=10, gamma=1)

    train_model(model, criterion=cross_entropy, optimizer=sgd, scheduler=exp_lr_scheduler, dataloaders=dataloaders,
                dataset_sizes=dataset_sizes)
