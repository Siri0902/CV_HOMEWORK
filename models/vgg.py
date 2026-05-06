import torch
import torchvision.models as models


def get_vgg16(pretrained=True, num_classes=24, in_channels=3):
    model = models.vgg16(pretrained=pretrained)
    if in_channels != 3:
        model.features[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
    if num_classes != 1000:
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def get_vgg19(pretrained=True, num_classes=24, in_channels=3):
    model = models.vgg19(pretrained=pretrained)
    if in_channels != 3:
        model.features[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
    if num_classes != 1000:
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model