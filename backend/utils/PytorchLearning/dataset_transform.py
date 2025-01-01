import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ds = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(), # 将 PIL 图像或 NumPy 转换ndarray为FloatTensor. 并将图像的像素强度值缩放到范围 [0., 1.] 内
#     target_transform=Lambda(
#         lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
#     ) # 创建一个大小为 10 的零张量 .scatter_() 在 维度 0 上  将标签 y 所对应的索引位置赋值为 value
# )

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# 创建个三维张量
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# 将取 3 张大小为 28x28 的图像作为样本小批量
input_image = torch.rand(3,28,28)
print(input_image.size())

# 1.将每个 2D 28x28 图像转换为 784 个像素值的连续数组
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# 2.使用其存储的权重和偏差对输入应用线性变换的模块
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# 3.在线性变换之后应用以引入非线性
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# 4.使用顺序容器来快速组合网络
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image) # logits 被缩放到 [0, 1] 的值，表示模型对每个类的预测概率

# 最后一层线性层返回logits
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# 神经网络中的许多层都是参数化的，即具有在训练期间优化的相关权重和偏差
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")