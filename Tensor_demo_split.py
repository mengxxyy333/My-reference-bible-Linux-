import torch

# chunk切片平均分，最后一片可能不足
a = torch.rand((3, 4))
print(a)
out = torch.chunk(a, 2, dim=0)
print(out)
out = torch.chunk(a, 2, dim=1)
print(out)

# split单独一个数字就平均切，给定一个列表就按列表切，尺寸对不上会报错
a = torch.rand((10, 4))
print(a)
out = torch.split(a, 3, dim=0)
print(out)

out = torch.split(a, [1, 3, 6], dim=0)
print(out)