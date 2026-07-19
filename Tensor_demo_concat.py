import torch

# cat在指定维度拼接,维度不会发生变化
a = torch.zeros((2, 4))
b = torch.ones((2, 4))
out = torch.cat((a, b), dim=0)
print(out)
out = torch.cat((a, b), dim=1)
print(out)

# stack按指定维度拼接，但是会产生新的维度
a = torch.linspace(1, 6, 6).view(2, 3)
b = torch.linspace(7, 12, 6).view(2, 3)
print(a)
print(b)
out = torch.stack((a, b), dim=0)
print(out)
out = torch.stack((a, b), dim=1)
print(out)
out = torch.stack((a, b), dim=2)
print(out)