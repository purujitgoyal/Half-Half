from torch import nn
from torchvision import models

from glore.gloreunit import GloreUnit
from model import Model


class GloreModel(Model):

    def get_model(self, finetune_conv=True, device="cpu"):
        model = self._get_resnet50()

        if not finetune_conv:
            for param in model.parameters():
                param.requires_grad = False

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
        model.fc = nn.Linear(in_features, self.out_features)
        model = model.to(device)

        return model


if __name__ == '__main__':
    gm = GloreModel()
    gm.get_model()
