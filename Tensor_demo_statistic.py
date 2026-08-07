import torch

'''
一些统计学相关函数
    均值、求和、求积
'''

# 默认整体求
a = torch.rand(2, 2)
print(a)
print(torch.mean(a)) # 均值
print(torch.sum(a)) # 求和
print(torch.prod(a)) # 求积

# 指定维度求,这样求可以降维
print(torch.mean(a, dim=0)) # 均值
print(torch.sum(a, dim=0)) # 求和
print(torch.prod(a, dim=0)) # 求积

print(torch.argmax(a, dim=0)) # 最大值在指定维度上的索引值
print(torch.argmin(a, dim=0)) # 最小值在指定维度上的索引值

print(torch.std(a)) # 标准差
print(torch.var(a)) # 方差
print(torch.median(a)) # 中数
print(torch.mode(a)) # 众数

# 直方图,统计某些区间的出现频次
a = torch.rand(2, 2) * 10
print(a)
print(torch.histc(a, 6, 0, 0)) # 分成六份，最大值与最小值写0即a中的最大值与最小值，计算出后将该区间均分六份

# bincount,只能处理一维张量,统计某些值的出现频次
a = torch.randint(0, 10, [10]) # 值在[0, 10)，10维张量
print(a)
print(torch.bincount(a))
# 可以统计某一类别样本的个数