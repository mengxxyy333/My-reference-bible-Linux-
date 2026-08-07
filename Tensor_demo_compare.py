import torch

'''
比较运算
    大于、小于、大于等于、小于等于、相等、不等、排序、第k大
'''

a = torch.rand(2, 3)
b = torch.rand(2, 3)
print(a)
print(b)

print(torch.eq(a, b)) # 是否相同，对于每个位置，给出一个结果张量
print(torch.equal(a, b)) # 是否相同，直接给出T或者F

print(torch.ge(a, b)) # 大于等于
print(torch.gt(a, b)) # 大于
print(torch.le(a, b)) # 小于等于
print(torch.lt(a, b)) # 小于
print(torch.ne(a, b)) # 不等于

# 形状不同会报错
# a = torch.rand(2, 3)
# b = torch.rand(2, 4)
# print(torch.eq(a, b))

# 排序
a = torch.tensor([1, 4, 4, 3, 5])
print(torch.sort(a))
print(torch.sort(a, descending=True)) # 降序

# 指定维度排序
a = torch.tensor([[1, 4, 4, 3, 5], [2, 3, 1, 3, 5]])
print(a.shape)
print(torch.sort(a))
print(torch.sort(a, dim=1, descending=True))

# topk，指定维度第k大的数字
a = torch.tensor([[2, 4, 3, 1, 5], [2, 3, 5, 1, 4]])
print(a.shape)
print(torch.topk(a, k=2, dim=1))

# 指定维度第k小的数字
print(torch.kthvalue(a, k=2, dim=0))
print(torch.kthvalue(a, k=2, dim=1))

# 有界性判断、inf、nan
a = torch.rand(2, 3)
print(a)
print(torch.isfinite(a))
print(torch.isfinite(a / 0))
print(torch.isinf(a / 0))
print(torch.isnan(a))

import numpy as np
a = torch.tensor([1, 2, np.nan])
print(torch.isnan(a))