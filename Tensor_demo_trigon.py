import torch

'''
三角函数同样支持in-place操作
'''

a = torch.zeros(2, 3)
b = torch.cos(a)
print(a)
print(b)