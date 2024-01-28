import timm
import torch
from torch import nn


class EmotionEstimatorModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionEstimatorModel, self).__init__()

        self.base_model = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        original_weights = self.base_model.conv_stem.weight
        self.base_model.conv_stem = torch.nn.Conv2d(1, self.base_model.conv_stem.out_channels, kernel_size=1, stride=1,
                                                    padding=0, bias=False)
        self.base_model.conv_stem.weight.data = torch.sum(original_weights, dim=1, keepdim=True)
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)



class EmotionBinEstimatorModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionBinEstimatorModel, self).__init__()

        self.base_model = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        original_weights = self.base_model.conv_stem.weight
        self.base_model.conv_stem = torch.nn.Conv2d(in_channels=1, # потому что принимаем чб картинку
                                                    out_channels=self.base_model.conv_stem.out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    bias=False)
        self.base_model.conv_stem.weight.data = torch.sum(original_weights, dim=1, keepdim=True)
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


class AgeEstimatorModel(nn.Module):
    def __init__(self):
        super(AgeEstimatorModel, self).__init__()
        self.base_model = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, 1)

    def forward(self, x):
        return self.base_model(x)


class RaceEstimatorModel(nn.Module):
    def __init__(self, num_classes):
        super(RaceEstimatorModel, self).__init__()
        self.base_model = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


class SexEstimatorModel(nn.Module):
    def __init__(self):
        super(SexEstimatorModel, self).__init__()
        self.base_model = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, 1)

    def forward(self, x):
        x = self.base_model(x)
        return torch.sigmoid(x).squeeze()  # Используйте сигмоиду для бинарной классификации
