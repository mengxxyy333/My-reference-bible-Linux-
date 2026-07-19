import torch

'''
广播机制:不同维度的张量进行运算
    每个张量必须至少有一个维度
    右对齐:低维度对齐高纬度最右侧的维度,前面补1,只有对应维度要么是1、要么相同,才可以运算,不然报错
'''

# 不同维度满足右对齐的运算，结果是用b去加a的每一行
a = torch.rand(2, 3)
b = torch.rand(3)
c = a + b
print(a)
print(b)
print(c)
print(c.shape)

# 会报错的情况
# a = torch.rand(2, 3)
# b = torch.rand(2)
# c = a + b
# print(a)
# print(b)
# print(c)
# print(c.shape)

# 更复杂的情况
a = torch.rand(2, 1, 1, 3)
b = torch.rand(4, 2, 3)
c = a + b
print(a)
print(b)
print(c)
print(c.shape)