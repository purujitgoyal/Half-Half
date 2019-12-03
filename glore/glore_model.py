import torch
from torch import nn

from .gloreunit import GloreUnit
from model import Model


class GloreModel(Model):

    def get_model(self, out_features=78, finetune_conv=True, device="cpu", model_wts=None):
        model = self._get_resnet50()

        # print(model)
        res4 = model.layer4
        res4_new = nn.Sequential(
            res4[0],
            GloreUnit(2048, 512),
            res4[1],
            GloreUnit(2048, 512),
            res4[2],
            GloreUnit(2048, 512)
        )

        model.layer4 = res4_new
        in_features = model.fc.in_features
        if model_wts is not None:
            model.fc = nn.Linear(in_features, 78)
            model.load_state_dict(torch.load(model_wts))

        if not finetune_conv:
            for param in model.parameters():
                param.requires_grad = False

        model.fc = nn.Linear(in_features, out_features)
        model = model.to(device)

        return model


if __name__ == '__main__':
    gm = GloreModel()
    gm.get_model()
