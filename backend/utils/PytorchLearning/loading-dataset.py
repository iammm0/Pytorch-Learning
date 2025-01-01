import matplotlib as plt
import torch
from matplotlib.pyplot import figure as fig
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

# 创建一个800 x 800的窗口
figure = plt.pyplot.figure(figsize=(8, 8))
# 设定3 x 3的窗口参数
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    # 从 0 到 len(training_data) - 1 中随机选取一个索引
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx] # 调取随机选取的索引所对应的图像与标签
    figure.add_subplot(rows, cols, i) # 将当前 figure 图形放置在指定窗格
    plt.pyplot.title(labels_map[label]) # 添加标题
    plt.pyplot.axis("off") # 隐藏坐标轴
    plt.pyplot.imshow(img.squeeze(), cmap="gray") # 去除多余维度 并以灰度的形式显示图像

# 显示绘制好的 figure
plt.pyplot.show()

# 使用 Dataloader 加载训练集与测试集的数据
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 迭代训练集中的特征与标签 赋值给 train_features 和 train_labels
train_features, train_labels = next(iter(train_dataloader))

# 输出这些特征和标签的 size 属性
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]

plt.pyplot.imshow(img, cmap="gray")
plt.pyplot.show()

print(f"Label: {label}")