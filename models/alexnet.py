import torch
import torchvision.models as models


def get_alexnet(pretrained=True, num_classes=24, in_channels=3):
    """
    修改AlexNet以适配28x28小图像
    - 减小第一层卷积的kernel_size和stride
    - 减少池化层数量
    - 调整classifier的输入维度
    """
    model = models.alexnet(pretrained=pretrained)
    
    # 对于28x28的小图像，需要大幅修改features部分
    # 原始AlexNet: Conv11-s4 -> Pool -> Conv5 -> Pool -> Conv3 -> Pool -> Conv3 -> Pool -> Conv3 -> Pool
    # 修改后: Conv3-s1 -> Pool -> Conv3 -> Pool -> Conv3 -> (移除最后2个pool)
    
    new_features = torch.nn.Sequential(
        # 第一层：减小kernel和stride
        torch.nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 28->14
        
        # 第二层
        torch.nn.Conv2d(64, 192, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 14->7
        
        # 第三层
        torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        
        # 第四层
        torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        
        # 第五层（移除最后的pool）
        torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        # 不再使用MaxPool2d，保持7x7的尺寸
    )
    
    model.features = new_features
    
    # 重新计算classifier的输入维度: 256 * 7 * 7 = 12544
    with torch.no_grad():
        dummy_input = torch.zeros(1, in_channels, 28, 28)
        dummy_output = model.features(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)
    
    # 重建classifier
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(),
        torch.nn.Linear(flattened_size, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(4096, num_classes),
    )
    
    # 移除avgpool
    model.avgpool = torch.nn.Identity()
    
    return model