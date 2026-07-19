import torch

# torch.where(condition, x, y),对于每个位置，满足condition输出x的,否则输出y的
a = torch.rand(4, 4)
b = torch.rand(4, 4)
print(a)
print(b)
out = torch.where(a > 0.5, a, b)
print(out)

# torch.index_select(input, dim, index),打印input的指定维度上的index对应向量,拿的是整个向量
a = torch.rand(4, 4)
print(a)
out = torch.index_select(a, dim=0, index=torch.tensor([0, 3, 2]))
print(out)

# torch.gather(input, dim, index),打印input的指定维度上的index索引对应的值,拿的是由索引值指定的一个值
# 可以这样理解:
# dim=0的情况下,out[x, y, z] = input[index[x, y, z], y, z]
# dim=1的情况下,out[x, y, z] = input[x, index[x, y, z], z]
# dim=2的情况下,out[x, y, z] = input[x, y, index[x, y, z]]
a = torch.linspace(1, 16, 16).view(4, 4)
print(a)
out = torch.gather(a, dim=0, index=torch.tensor([[0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 3, 3]]))
print(out)

# torch.masked_index(input, index),输出那些符合条件的元素
a = torch.linspace(1, 16, 16).view(4, 4)
mask = torch.gt(a, 8)
print(a)
print(mask)
out = torch.masked_select(a, mask)
print(out)

# torch.take(input, index),拿出指定索引的元素，会将input拉直后索引
a = torch.linspace(1, 16, 16).view(4, 4)
out = torch.take(a, index=torch.tensor([0, 15, 13, 10]))
print(out)

# torch.nonzero(input),打印所有非0元素的索引
a = torch.tensor([[0, 1, 2, 0], [2, 3, 0, 1]])
out = torch.nonzero(a)
print(out)