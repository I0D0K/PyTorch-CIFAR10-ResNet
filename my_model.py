import torch.nn as nn
import torch
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, n_chans1=32):
        super(ResNet, self).__init__()
        self.n_chans1 = n_chans1

        # 第一个卷积层，输入通道数为3（RGB图像），输出通道数为n_chans1，卷积核大小为3x3，padding为1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)

        # 第二个卷积层，输入通道数为n_chans1，输出通道数为n_chans1//2，卷积核大小为3x3，padding为1
        self.conv2 = nn.Conv2d(n_chans1, n_chans1//2, kernel_size=3, padding=1)

        # 第三个卷积层，输入和输出通道数都为n_chans1//2，卷积核大小为3x3，padding为1
        self.conv3 = nn.Conv2d(n_chans1//2, n_chans1//2, kernel_size=3, padding=1)

        # 全连接层1，输入大小为4*4*n_chans1//2，输出大小为32
        self.fc1 = nn.Linear(4 * 4 * n_chans1//2, 32)

        # 全连接层2，输入大小为32，输出大小为10（对应10个类别）
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # 第一个卷积层后接ReLU激活函数和最大池化层
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)

        # 第二个卷积层后接ReLU激活函数和最大池化层
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out1 = out

        # 第三个卷积层后接ReLU激活函数和最大池化层，同时加上第二个卷积层的输出（残差连接）
        out = F.max_pool2d(torch.relu(self.conv3(out)) + out1, 2)

        # 将输出展平为一维向量
        out = out.view(-1, 4 * 4 * self.n_chans1//2)

        # 全连接层1后接ReLU激活函数
        out = torch.relu(self.fc1(out))

        # 全连接层2，最终输出
        out = self.fc2(out)

        return out


class ResBlk(nn.Module):
    # resnet block

    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b,ch_in,h,w] => [b,ch_out,h,w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        # x: [b,ch,h,w]

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # short cut
        # extra module: [b, ch_in, h,w] => [b,ch_out,h,w]
        # element-wise add:
        out = self.extra(x) + out

        return out


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h ,w]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(128, 256, stride=2)
        # # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(256, 512, stride=2)
        # # [b, 512, h, w] => [b, 1024, h, w]
        self.blk4 = ResBlk(512, 512, stride=2)

        self.outlayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print('after conv:',x.shape)   #[b,512,2,2]

        # [b,512,h,w] => [b,512,1,1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)

        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x
