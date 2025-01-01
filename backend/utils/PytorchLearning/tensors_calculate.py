import torch
from torch.onnx.symbolic_opset9 import tensor

tensor1 = torch.ones(4, 4)
print(tensor1)
print(f"First row: {tensor1[0]}")
print(f"First column: {tensor1[:, 0]}")
print(f"Last column: {tensor1[:, -1]}")
tensor1[:,1] = 0
print(tensor1)

t1 = torch.cat([tensor1,tensor1,tensor1], dim = 0)
print(t1)

# 矩阵乘法
# y1 = tensor @ tensor.T
# y2 = tensor.matmul(tensor.T)

# y3 = torch.rand_like(y1)
# torch.matmul(tensor1, tensor1.T, out=y3)

# 逐元素乘积
# z1 = tensor * tensor
# z2 = tensor.mul(tensor)

# z3 = torch.rand_like(tensor1)
# torch.mul(tensor1, tensor1, out=z3)

tensor1 = torch.ones(5, 5)

tensor1[0, :] = 0

print(tensor1)

print(f"{tensor1} \n")
tensor1.add_(5)
print(tensor1)

agg = tensor1.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))