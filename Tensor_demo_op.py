import torch

'''
对于带有下划线的方法，是在运算结束后将结果再赋值给前面的张量
'''

# 加法
a = torch.rand(2, 3)
b = torch.rand(2, 3)
print(a)
print(b)
print("加法：")
print(a + b)
print(a.add(b))
print(torch.add(a, b))
print(a)
print(a.add_(b))
print(a)

# 减法
print("减法：")
print(a - b)
print(a.sub(b))
print(torch.sub(a, b))
print(a)
print(a.sub_(b))
print(a)

# 乘法
print("乘法：")
print(a * b)
print(a.mul(b))
print(torch.mul(a, b))
print(a)
print(a.mul_(b))
print(a)

# 除法
print("除法：")
print(a / b)
print(a.div(b))
print(torch.div(a, b))
print(a)
print(a.div_(b))
print(a)

# 矩阵乘法
a = torch.ones(2, 1)
b = torch.ones(1, 2)
print(a @ b)
print(a.matmul(b))
print(torch.matmul(a, b))
print(torch.mm(a, b))

# 高维矩阵，前面维度必须保持一致，在最后两个维度进行计算
a = torch.ones(1, 2, 3, 4)
b = torch.ones(1, 2, 4, 3)
print(a.matmul(b))
print(a.matmul(b).shape)

# 幂运算，将张量各数字求指定的幂
a = torch.tensor([1, 2])
print(torch.pow(a, 3));
print(a.pow(3))
print(a**3)
print(a.pow_(3))
print(a)

# 自然常数e的指定次幂
a = torch.tensor([1, 2], dtype=torch.float32)
print(a.type())
print(torch.exp(a))
print(torch.exp_(a))
print(a.exp())
print(a.exp_())

# 对数运算
a = torch.tensor([1, 2], dtype=torch.float32)
print(torch.log(a))
print(torch.log_(a))
print(a.log())
print(a.log_())

# 开平方
a = torch.tensor([10, 2], dtype=torch.float32)
print(torch.sqrt(a))
print(torch.sqrt_(a))
print(a.sqrt())
print(a.sqrt_())

# 像带有下划线的这种运算，不产生临时变量，叫做in-place操作，即“原地操作”