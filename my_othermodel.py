import torch
import torch.nn as nn
import torch.nn.functional as F

# 自定义残差块
class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=stride, padding=2)  # 使用5x5卷积核
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)  # 使用3x3卷积核
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.LeakyReLU(negative_slope=0.1)  # 使用LeakyReLU激活函数

        # 若输入输出通道数不同，添加卷积使通道数匹配
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 残差连接
        out = self.extra(x) + out

        return out


# 自定义ResNet模型
class ResNetCustom(nn.Module):
    def __init__(self):
        super(ResNetCustom, self).__init__()
        # 输入层：使用3x3卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1)
        )
        # 残差块
        self.blk1 = ResBlk(64, 128, stride=2)
        self.blk2 = ResBlk(128, 256, stride=2)
        self.blk3 = ResBlk(256, 512, stride=2)
        # 分类层
        self.outlayer = nn.Linear(512 * 1 * 1, 10)
        # Dropout层：为了防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)

        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, [1, 1])

        # 展平
        x = x.view(x.size(0), -1)

        # Dropout
        x = self.dropout(x)

        # 输出
        x = self.outlayer(x)
        return x







# 定义Bottleneck模块
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 残差连接（如果输入输出通道数不一致，使用1x1卷积调整维度）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # 残差连接

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity  # 添加残差
        out = F.relu(out)  # 最后的激活函数

        return out


# 定义ResNet50模型
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第一阶段（包含3个残差块）
        self.layer1 = self._make_layer(64, 256, 3, stride=1)
        # 第二阶段（包含4个残差块）
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        # 第三阶段（包含6个残差块）
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)
        # 第四阶段（包含3个残差块）
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(2048, 10)  # 输出10个类别
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 平均池化
        x = self.avgpool(x)
        # 展平
        x = torch.flatten(x, 1)
        # 全连接层输出
        x = self.fc(x)
        return x
