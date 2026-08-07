import torch

'''
a.clamp(l, r):将a的值裁剪到[l, r]之间
'''

a = torch.rand(2, 2) * 10
print(a)
a = a.clamp(2, 5)
print(a)