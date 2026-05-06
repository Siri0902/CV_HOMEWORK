import torch
import torchvision.models as models


def get_alexnet(pretrained=True, num_classes=24, in_channels=3):
    model = models.alexnet(pretrained=pretrained)
    if in_channels != 3:
        model.features[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2, bias=False)
    if num_classes != 1000:
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model