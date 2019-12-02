from abc import ABC, abstractmethod

from torchvision import models


class Model(ABC):

    def __init__(self, out_features=79):
        self.out_features = out_features
        super(Model, self).__init__()

    @staticmethod
    def _get_resnet50(pretrained=True):
        resnet50 = models.resnet50(pretrained=pretrained)
        return resnet50

    @staticmethod
    def _get_resnet101(pretrained=True):
        resnet101 = models.resnet101(pretrained=pretrained)
        return resnet101

    @abstractmethod
    def get_model(self, finetune_conv=False, device="cpu"):
        pass
