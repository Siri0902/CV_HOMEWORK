import torch.nn as nn

from .my_dnn import MyDNN
from .vgg import get_vgg16, get_vgg19
from .alexnet import get_alexnet
from .resnet import (
    get_resnet18, get_resnet34, get_resnet50,
    get_resnet101, get_resnet152
)


def get_model(model_name, pretrained=True, num_classes=24, in_channels=1):
    """模型工厂函数，根据名称返回对应的模型实例"""
    models_map = {
        'MyDNN': lambda: MyDNN(num_classes=num_classes, in_channels=in_channels),
        'VGG16': lambda: get_vgg16(pretrained=pretrained, num_classes=num_classes, in_channels=in_channels),
        'VGG19': lambda: get_vgg19(pretrained=pretrained, num_classes=num_classes, in_channels=in_channels),
        'ResNet18': lambda: get_resnet18(pretrained=pretrained, num_classes=num_classes, in_channels=in_channels),
        'ResNet34': lambda: get_resnet34(pretrained=pretrained, num_classes=num_classes, in_channels=in_channels),
        'ResNet50': lambda: get_resnet50(pretrained=pretrained, num_classes=num_classes, in_channels=in_channels),
        'ResNet101': lambda: get_resnet101(pretrained=pretrained, num_classes=num_classes, in_channels=in_channels),
        'ResNet152': lambda: get_resnet152(pretrained=pretrained, num_classes=num_classes, in_channels=in_channels),
        'AlexNet': lambda: get_alexnet(pretrained=pretrained, num_classes=num_classes, in_channels=in_channels),
    }

    if model_name not in models_map:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models_map.keys())}")

    return models_map[model_name]()


__all__ = [
    'MyDNN', 'get_vgg16', 'get_vgg19', 'get_alexnet',
    'get_resnet18', 'get_resnet34', 'get_resnet50', 'get_resnet101', 'get_resnet152',
    'get_model'
]