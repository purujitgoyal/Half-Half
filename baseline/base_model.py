import torch.nn as nn

from model import Model


class BaseModel(Model):

    def get_model(self, finetune_conv=False, device="cpu"):
        model = self._get_resnet50()

        if not finetune_conv:
            for param in model.parameters():
                param.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.out_features)
        model = model.to(device)
        return model

