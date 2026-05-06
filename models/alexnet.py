import torch
import torchvision.models as models


def get_alexnet(pretrained=True, num_classes=24, in_channels=3):
    """
    AlexNet标准结构（输入224×224, 3通道）
    """
    model = models.alexnet(pretrained=pretrained)
    
    if in_channels != 3:
        model.features[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2)
    
    if num_classes != 1000:
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    
    return model
