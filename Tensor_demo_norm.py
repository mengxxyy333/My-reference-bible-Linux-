import torch

'''
计算p-范数距离:dist
计算张量的范数:norm,0范数指非0元素个数,1范数指元素绝对值的和
'''

a = torch.rand(1, 1)
b = torch.rand(1, 1)
print(a, b)
print(torch.dist(a, b, p=1)) # a-b绝对值各项求和，l1距离
print(torch.dist(a, b, p=2)) # a-b各项平方求和开根号，l2距离
print(torch.dist(a, b, p=3)) # a-b各项三次幂求和开三次方根，l3距离

print(torch.norm(a))
print(torch.norm(a, p=1))
print(torch.norm(a, p = 'fro')) # 核范数