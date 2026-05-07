from .alexnet import get_alexnet
from .my_dnn import SimpleCNN, EnhancedCNN
from .resnet import (
    get_resnet18, get_resnet34, get_resnet50,
    get_resnet101, get_resnet152
)
from .vgg import get_vgg16, get_vgg19


# 标准模型需要224x224输入和3通道
STANDARD_MODELS = {'AlexNet', 'VGG16', 'VGG19', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'}
# 自定义模型使用28x28输入和1通道
CUSTOM_MODELS = {'SimpleCNN', 'EnhancedCNN'}


def get_model_input_config(model_name):
    """获取模型所需的输入配置
    
    Returns:
        tuple: (target_size, in_channels, convert_to_rgb)
    """
    if model_name in STANDARD_MODELS:
        return 64, 3, True
    elif model_name in CUSTOM_MODELS:
        return 28, 1, False
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_model(model_name, pretrained=True, num_classes=24, in_channels=None):
    """Model factory function that returns model instance based on name"""
    # 自动推断in_channels（如果未指定）
    if in_channels is None:
        _, in_channels, _ = get_model_input_config(model_name)
    
    models_map = {
        'SimpleCNN': lambda: SimpleCNN(num_classes=num_classes, in_channels=in_channels),
        'EnhancedCNN': lambda: EnhancedCNN(num_classes=num_classes, in_channels=in_channels),
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
    'SimpleCNN', 'EnhancedCNN',
    'get_vgg16', 'get_vgg19', 'get_alexnet',
    'get_resnet18', 'get_resnet34', 'get_resnet50', 'get_resnet101', 'get_resnet152',
    'get_model'
]
