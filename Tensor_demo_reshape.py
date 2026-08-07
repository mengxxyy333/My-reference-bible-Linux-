import torch

a = torch.rand(2, 3)
print(a)

# reshape(),这个变形可以理解为:将原张量拉成一个向量，再按指定维度拼接
out = torch.reshape(a, (3, 2))
print(out)

# t(),二维转置，行变列，列变行
print(torch.t(out))

# transpose() 交换指定的两个维度,两个维度就是行变列,列变行
print(torch.transpose(out, 0, 1))
# 高维的例子
a = torch.rand(1, 2, 3)
print(a)
out = torch.transpose(a, 0, 1)
print(out)
print(out.shape)

# squeeze(),删除维度是1的维度
out = torch.squeeze(a)
print(out)
print(out.shape)
# unsqueeze(),可以在指定维度上扩展,扩展出1
out = torch.unsqueeze(a, -1)
print(out.shape)

# unbind(),用于删除指定维度
out = torch.unbind(a, dim=2)
print(out)

# flip(),翻转指定维度
print(a)
print(torch.flip(a, dims=[1, 2])) # 此处传入列表，按顺序翻转

# rot90(),逆时针翻转90度,默认1次，可以指定次数,-1是顺时针旋转
# 也可以指定维度列表进行旋转
print(a)
out = torch.rot90(a, 2)
print(out)
print(out.shape)