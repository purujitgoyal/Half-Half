from gcn.gcnunit import GCNResnet
from model import Model
from .gcnunit import GCNResnet


class GcnModel(Model):

    def get_model(self, t, adj_file=None, in_channel=300, out_features=80, finetune_conv=False, device="cpu"):
        model = self._get_resnet50(pretrained=True)

        model = GCNResnet(model, out_features, finetune_conv=finetune_conv, t=t, adj_file=adj_file, in_channel=in_channel)
        model = model.to(device)

        return model


if __name__ == '__main__':
    gcn = GcnModel()
    gcn.get_model()
