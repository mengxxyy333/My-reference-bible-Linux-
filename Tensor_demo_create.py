import torch

'''以下是几种常用的张量'''

# 直接初始化的方式定义Tensor
a = torch.Tensor([[1, 2], [3, 4]])
print(a)
print(a.type())

# 指定形状创建Tensor，初始化值是随机数，即内存中的值
a = torch.Tensor(2, 3)
print(a)
print(a.type())

# 创建全1张量
a = torch.ones(2, 2)
print(a)
print(a.type())

# 创建全0张量
a = torch.zeros(2, 2)
print(a)
print(a.type())

# 创建主对角线全1张量
a = torch.eye(2, 3)
print(a)
print(a.type())

# 创建一个全0、全1张量，形状与指定张量相同
b = torch.Tensor(2, 3)
b = torch.zeros_like(b)
print(b)
print(b.type())
b = torch.ones_like(b)
print(b)
print(b.type())


'''以下是随机张量'''

a = torch.rand(2, 2)
print(a)
print(a.type())

# 满足某一分布的随机张量
# 正态分布
# 生成的张量这样理解：五组均值与标准差可以对应五个正态分布的函数，用该函数生成的随机数值
a = torch.normal(mean=0.0, std=torch.rand(5))
print(a)
print(a.type())

# 均匀分布，值在[-1, 1]之间
a = torch.Tensor(2, 2).uniform_(-1, 1)
print(a)
print(a.type())


'''以下是序列张量'''

# 范围左闭右开，加步长
a = torch.arange(0, 10, 1)
print(a)
print(a.type())

a = torch.arange(0, 10, 2)
print(a)
print(a.type())

# 左闭右闭区间拿到k个数字，这k个数字之间等间隔
a = torch.linspace(2, 10, 4)
print(a)
print(a.type())

# 生成一个打乱的序列，左闭右开
a = torch.randperm(10)
print(a)
print(a.type())


'''以下是tensor与numpy的对比'''
# numpy与tensor有许多相似的地方，可以相互转换
# 例如图像处理会将图片的特征像素值读入为numpy数组，再转换为tensor送入模型
import numpy as np
a = np.array([[1, 2], [2, 3]])
print(a)