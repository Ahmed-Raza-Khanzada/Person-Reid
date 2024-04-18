from torch import nn
import torch.nn.functional as F
import timm


class Model(nn.Module):

    def __init__(self,embed_size = 512 ,model_name= "efficientnet_b0"):
        super(Model, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.classifier = nn.Linear(in_features = self.model.classifier.in_features, out_features =embed_size)
    def forward(self, x):
        return self.model(x)