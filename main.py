import datetime
from my_othermodel import *
from my_model import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import time
# 转换为张量并归一化到[-1, 1]
transform_1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为PyTorch张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 随机旋转角度，增强模型对不同旋转角度的适应能力
transform_2 = transforms.Compose([
    transforms.RandomRotation(30),  # 随机旋转 ±30度
    transforms.ToTensor(),  # 转换为PyTorch张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 使用随机裁剪和颜色抖动的组合
# 适合对图像的局部变化和光照变化进行增强
transform_3 = transforms.Compose([
    transforms.RandomResizedCrop(32),  # 随机裁剪并调整大小到32x32
    transforms.ColorJitter(  # 颜色抖动，模拟不同光照条件
        brightness=0.5,  # 随机调整亮度
        contrast=0.5,    # 随机调整对比度
        saturation=0.5,  # 随机调整饱和度
        hue=0.5          # 随机调整色调
    ),
    transforms.ToTensor(),  # 转换为PyTorch张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])
# 综合多种增强方法的组合
# 适用于需要处理复杂数据变化的场景
transform_4 = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(20),  # 随机旋转 ±20度
    transforms.ColorJitter(  # 颜色抖动
        brightness=0.2,  # 随机调整亮度
        contrast=0.2,    # 随机调整对比度
        saturation=0.2   # 随机调整饱和度
    ),
    transforms.ToTensor(),  # 转换为PyTorch张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])
writer = SummaryWriter('logs')
# 类名 这是CIFAR-10数据集的类别名称，对应着数据集中的10个不同类别。
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
# 这段代码定义了一个数据预处理管道，将图像数据转换为张量，并进行标准化。
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载训练集,这里使用TorchVision库中的CIFAR10类加载CIFAR-10训练集，将其进行预处理并使用DataLoader封装成一个可以迭代的数据加载器，以便在训练时批量获取数据。
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                           shuffle=True)

# 下载并加载测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64,
                                          shuffle=False)
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def training_loop(n_epochs, optimizer, loss_fn, model, train_loader,val_loader):
    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        train_loss = 0.0
        model.train()  # 设置模型为训练模式

        # Training step
        for images, labels in train_loader:
            images = images.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation step
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # 禁用梯度计算
            for images, labels in val_loader:
                images = images.to(device=device)
                labels = labels.to(device=device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total

        end_time = time.time()
        epoch_time = end_time - start_time

        # 输出训练和验证信息
        print(
            '{} Epoch {}, Training loss: {:.4f}, Validation loss: {:.4f}, Validation accuracy: {:.2f}, Epoch time: {:.2f} seconds'.format(
                datetime.datetime.now(), epoch, train_loss, val_loss, val_accuracy, epoch_time
            ))

        # 使用 SummaryWriter 记录训练和验证指标
        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

    writer.close()


def validate(model, train_loader, val_loader):
    accdict = {}
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name, correct / total))
        accdict[name] = correct / total
    return accdict

# model = ResNet(n_chans1=32).to(device=device)
model = ResNetCustom().to(device=device)
#model = ResNet18().to(device=device)
#model = ResNet50().to(device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

total_params = count_parameters(model)
print(f'Total number of parameters in the model: {total_params}')

writer = SummaryWriter('logs')
training_loop(
    n_epochs=10,
    optimizer=optimizer,
    model=model,
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=test_loader,
)
torch.save(model, 'cifar-10-1')
validate(model, train_loader, test_loader)
