
# Define your model architecture

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale


class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion, use_se, activation):
        super(MobileNetV3Block, self).__init__()
        self.use_se = use_se
        self.stride = stride
        hidden_dim = in_channels * expansion

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = activation()

        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                padding=kernel_size // 2, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.act2 = activation()

        self.se = SEBlock(hidden_dim) if use_se else nn.Identity()

        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.dwconv(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.se(out)

        out = self.conv2(out)
        out = self.bn3(out)

        if self.use_res_connect:
            return identity + out
        else:
            return out


class EfficientMobileNetV3(nn.Module):
    def __init__(self, num_classes=100):
        super(EfficientMobileNetV3, self).__init__()
        activation = nn.Hardswish

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),  
            nn.BatchNorm2d(16),
            activation()
        )

        self.blocks = nn.Sequential(
            MobileNetV3Block(16, 16, 3, 1, 1, False, nn.ReLU),   
            MobileNetV3Block(16, 24, 3, 2, 4, False, nn.ReLU),   
            MobileNetV3Block(24, 24, 3, 1, 3, False, nn.ReLU),
            MobileNetV3Block(24, 40, 5, 2, 3, True, activation),
            MobileNetV3Block(40, 40, 5, 1, 3, True, activation),
            MobileNetV3Block(40, 80, 3, 2, 6, False, activation),
            MobileNetV3Block(80, 112, 3, 1, 6, True, activation),
            MobileNetV3Block(112, 160, 5, 2, 6, True, activation), 
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(160, 960, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(960),
            activation()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



# Function to load the saved model
def load_model(MODEL_PATH, num_classes=100):
    # Initialize the model architecture
    model = EfficientMobileNetV3(num_classes=num_classes)
    
    # Load the model weights
    import torch

    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    
    # Return the loaded model
    return model


