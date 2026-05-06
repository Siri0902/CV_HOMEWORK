import torch
import torchvision.models as models


def get_resnet18(pretrained=True, num_classes=24, in_channels=3):
    model = models.resnet18(pretrained=pretrained)
    if in_channels != 3:
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if num_classes != 1000:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet34(pretrained=True, num_classes=24, in_channels=3):
    model = models.resnet34(pretrained=pretrained)
    if in_channels != 3:
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if num_classes != 1000:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet50(pretrained=True, num_classes=24, in_channels=3):
    model = models.resnet50(pretrained=pretrained)
    if in_channels != 3:
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if num_classes != 1000:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet101(pretrained=True, num_classes=24, in_channels=3):
    model = models.resnet101(pretrained=pretrained)
    if in_channels != 3:
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if num_classes != 1000:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet152(pretrained=True, num_classes=24, in_channels=3):
    model = models.resnet152(pretrained=pretrained)
    if in_channels != 3:
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if num_classes != 1000:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model