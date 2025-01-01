import numpy as np
import torch
from torch.onnx.symbolic_opset9 import tensor

# 直接从数据创建
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# 从numpy数组创建
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print("\n")

print(x_data)
print(x_np)

# 保持了x_data的属性
x_ones = torch.ones_like(x_data)
# 重构了x_data的数据类型位浮点数
x_rand = torch.rand_like(x_data, dtype=torch.float)

print(f"Ones Tensor: \n {x_ones} \n")
print(f"Random Tensor: \n {x_rand} \n")


# 通过定义张量的形状来创建特殊的张量
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# 张量的属性

tensor1 = torch.rand(3, 4)

print(f"Shape of tensor1: \n {tensor1.shape} \n")
print(f"Datatype of tensor1: \n {tensor1.dtype} \n")
print(f"Device tensor is stored on: \n {tensor1.device} \n")

if torch.cuda.is_available():
    tensor1 = tensor.to("cuda")
    print(f"Device tensor is stored on: \n {tensor1.device} \n")
