import torch

'''
随机抽样
'''

torch.manual_seed(1) # 设置随机种子,为了每次执行抽样结果一致

mean = torch.rand(1, 2)
std = torch.rand(1, 2)
print(torch.normal(mean, std))