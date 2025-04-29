
import torch.nn.functional as F

import torch
import torch.nn as nn
from torchvision import models
#简单的多尺度卷积网络
#简单的多尺度卷积网络
class CNN_MultiScale(nn.Module):
    def __init__(self, num_classes):
        super(CNN_MultiScale, self).__init__()

        # 多尺度卷积分支
        self.branch3x3 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 小感受野
        self.branch5x5 = nn.Conv2d(3, 16, kernel_size=5, padding=2)  # 中等感受野
        self.branch7x7 = nn.Conv2d(3, 16, kernel_size=7, padding=3)  # 大感受野

        # 融合后通道数为 16*3=48，继续普通卷积处理
        self.conv2 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 输入图像是 224x224，3次pool之后为 28x28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 多尺度并行卷积
        x1 = F.relu(self.branch3x3(x))
        x2 = F.relu(self.branch5x5(x))
        x3 = F.relu(self.branch7x7(x))

        # 通道维度拼接: [B, 48, H, W]
        x = torch.cat([x1, x2, x3], dim=1)

        x = self.pool(F.relu(self.conv2(x)))   # -> [B, 64, 112, 112]
        x = self.pool(F.relu(self.conv3(x)))   # -> [B, 128, 56, 56]
        x = self.pool(x)                       # -> [B, 128, 28, 28]

        x = x.view(-1, 128 * 28 * 28)          # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        # 使用预训练的ResNet18, 使用 weights 参数
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 替换最后一层为自己的全连接层，类别数为 num_classes
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),  # 减少维度，降低模型复杂度
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x=self.base_model(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        # 加载预训练的 ResNet50 模型
        self.base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # 替换最后一层为我们自己的全连接层，类别数为 num_classes
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),  # 减少维度，降低模型复杂度
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        # 加载预训练的 ResNet50 模型
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # 替换最后一层为我们自己的全连接层，类别数为 num_classes
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),  # 减少维度，降低模型复杂度
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

