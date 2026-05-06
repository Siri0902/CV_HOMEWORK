import torch
import torchvision.models as models


def get_resnet18(pretrained=True, num_classes=24, in_channels=3):
    """
    修改ResNet18以适配28x28小图像
    - 减小第一层卷积的kernel_size和stride，避免过度下采样
    """
    model = models.resnet18(pretrained=pretrained)
    
    # 对于28x28的小图像，修改第一层卷积：
    # 原始: kernel=7, stride=2, padding=3 -> 28/2 = 14
    # 修改: kernel=3, stride=1, padding=1 -> 28/1 = 28
    if in_channels != 3:
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    else:
        # 即使是3通道，也需要修改以适应小图像
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 移除第一个maxpool层，防止进一步下采样
    model.maxpool = torch.nn.Identity()
    
    if num_classes != 1000:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet34(pretrained=True, num_classes=24, in_channels=3):
    """
    修改ResNet34以适配28x28小图像
    """
    model = models.resnet34(pretrained=pretrained)
    
    if in_channels != 3:
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    else:
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    model.maxpool = torch.nn.Identity()
    
    if num_classes != 1000:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet50(pretrained=True, num_classes=24, in_channels=3):
    """
    修改ResNet50以适配28x28小图像
    """
    model = models.resnet50(pretrained=pretrained)
    
    if in_channels != 3:
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    else:
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    model.maxpool = torch.nn.Identity()
    
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