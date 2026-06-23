from matplotlib import pyplot as plt
import torch
import torchvision

cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
torch.set_printoptions(edgeitems=2, linewidth=75)
torch.manual_seed(123)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

num_images_per_class = 5  # 每个类别展示的图片数量
num_classes = 10

fig = plt.figure(figsize=(8 * num_images_per_class, 3))
for i in range(num_classes):
    # 筛选出当前类别i的所有图片索引
    indices = [index for index, (_, label) in enumerate(cifar10) if label == i]
    for j in range(num_images_per_class):
        # 依次选择对应索引
        index = indices[j]
        img, _ = cifar10[index]
        ax = fig.add_subplot(num_images_per_class, num_classes, j * num_classes + 1 + i, xticks=[], yticks=[])
        ax.set_title(class_names[i])
        plt.imshow(img)
plt.show()
