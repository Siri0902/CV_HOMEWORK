import torch
import torchvision.models as models


def get_vgg16(pretrained=True, num_classes=24, in_channels=3):
    """
    修改VGG16以适配28x28小图像
    - 移除最后2个MaxPool层及其对应的卷积层，防止特征图尺寸过小
    - 调整classifier的输入维度
    """
    model = models.vgg16(pretrained=pretrained)
    
    # 修改第一层卷积以适配输入通道数
    if in_channels != 3:
        model.features[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
    
    # 对于28x28输入，需要移除最后的池化层以防止特征图变为0
    # 原始VGG16 features有31层，包含5个MaxPool
    # 28x28 -> pool1: 14x14 -> pool2: 7x7 -> pool3: 3x3 -> pool4: 1x1 (太小!)
    # 所以我们只保留到pool3，即保留前23层 (0-22)
    new_features = torch.nn.Sequential(*list(model.features.children())[:23])
    model.features = new_features
    
    # 重新计算classifier的输入维度
    # 28x28 -> 经过3次pooling (每次/2) -> 28/8 = 3.5 -> 向下取整为3x3
    # 输出通道数为512，所以是 512 * 3 * 3 = 4608
    with torch.no_grad():
        dummy_input = torch.zeros(1, in_channels, 28, 28)
        dummy_output = model.features(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)
        print(f"VGG16 features output shape: {dummy_output.shape}, flattened size: {flattened_size}")
    
    # 重建classifier
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(flattened_size, 4096),
        torch.nn.ReLU(True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, num_classes),
    )
    
    # 移除avgpool，因为我们已经在features中控制了尺寸
    model.avgpool = torch.nn.Identity()
    
    return model


def get_vgg19(pretrained=True, num_classes=24, in_channels=3):
    """
    修改VGG19以适配28x28小图像
    - 移除最后2个MaxPool层及其对应的卷积层，防止特征图尺寸过小
    - 调整classifier的输入维度
    """
    model = models.vgg19(pretrained=pretrained)
    
    # 修改第一层卷积以适配输入通道数
    if in_channels != 3:
        model.features[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
    
    # VGG19与VGG16类似，也只保留前3个pooling层
    # VGG19 features有37层，保留前23层
    new_features = torch.nn.Sequential(*list(model.features.children())[:23])
    model.features = new_features
    
    # 重新计算classifier的输入维度
    with torch.no_grad():
        dummy_input = torch.zeros(1, in_channels, 28, 28)
        dummy_output = model.features(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)
        print(f"VGG19 features output shape: {dummy_output.shape}, flattened size: {flattened_size}")
    
    # 重建classifier
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(flattened_size, 4096),
        torch.nn.ReLU(True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, num_classes),
    )
    
    # 移除avgpool
    model.avgpool = torch.nn.Identity()
    
    return model