from torchvision import models
import torch.nn as nn


def _get_resnet50(pretrained=True):
    resnet50 = models.resnet50(pretrained=pretrained)
    return resnet50


def get_model(out_features, finetune_conv=False, device="cpu"):
    model = _get_resnet50()

    if not finetune_conv:
        for param in model.parameters():
            param.requires_grad = False
    # Todo: experiment with finetune_conv = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, out_features)
    model = model.to(device)
    return model


# if __name__ == '__main__':
#     print(_get_resnet50())
