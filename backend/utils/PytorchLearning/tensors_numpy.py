# CPU 和 Numpy 数组上的张量可以
# 共享其底层内存位置，改变其中一个也会改变另一个
# ！！！！
import torch
import numpy as np

# 张量转数组
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# 数组转张量
np.add(n, 1, out=n)

print(f"t: {t}")
print(f"n: {n}")