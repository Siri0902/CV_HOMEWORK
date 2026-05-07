import torch
import torch.nn as nn


def get_activation(activation_name):
    """Get activation function by name"""
    activation_name = activation_name.lower()
    if activation_name == 'relu':
        return nn.ReLU(inplace=True)
    elif activation_name == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation_name == 'gelu':
        return nn.GELU()
    elif activation_name == 'elu':
        return nn.ELU()
    else:
        return nn.ReLU(inplace=True)


class SimpleCNN(nn.Module):
    """
    Simple CNN Architecture (Section 2.2)
    Basic CNN with Conv2D, MaxPool, and Dense layers
    """

    def __init__(self, in_channels=1, num_classes=24):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # First Conv Block
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second Conv Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class EnhancedCNN(nn.Module):
    """
    Enhanced CNN Architecture with ResNet-style blocks (Section 2.3)
    Uses residual connections for better gradient flow
    """

    def __init__(self, in_channels=1, num_classes=24, activation='ReLU'):
        super(EnhancedCNN, self).__init__()
        self.activation_fn = get_activation(activation)

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Residual Block 1
        self.res_block1 = self._make_residual_block(64, 128, activation)

        # Residual Block 2
        self.res_block2 = self._make_residual_block(128, 256, activation)

        # Residual Block 3
        self.res_block3 = self._make_residual_block(256, 512, activation)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            get_activation(activation),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _make_residual_block(self, in_channels, out_channels, activation):
        """Create a residual block with skip connection"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            get_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            get_activation(activation),
            nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity(),
        )

    def forward(self, x):
        x = self.conv1(x)

        # Residual Block 1 with skip connection
        identity = x
        x = self.res_block1[:6](x)
        if identity.shape != x.shape:
            identity = self.res_block1[6](identity)
        x = x + identity
        x = nn.functional.relu(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # Residual Block 2 with skip connection
        identity = x
        x = self.res_block2[:6](x)
        if identity.shape != x.shape:
            identity = self.res_block2[6](identity)
        x = x + identity
        x = nn.functional.relu(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # Residual Block 3 with skip connection
        identity = x
        x = self.res_block3[:6](x)
        if identity.shape != x.shape:
            identity = self.res_block3[6](identity)
        x = x + identity
        x = nn.functional.relu(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.avgpool(x)
        x = self.classifier(x)
        return x
