import torch

'''
每个tensor都有三个属性:dtype、device(放置在什么设备)、layout(稀疏或稠密存储)
'''

# 默认都是稠密的
dev = torch.device("cuda")
a = torch.tensor([2, 2], dtype=torch.float32, device=dev)
print(a)

# 定义稀疏张量，需要传入非0元素坐标与值
# 这里对于坐标张量会有警告，原因是默认关闭了该坐标的检查，检查内容包括：从小到大排序、不能出现同一坐标
# 开启检查会导致速度稍微慢一点，但是更安全。开启检查代码如下：
# torch.sparse.check_sparse_tensor_invariants.enable()
i = torch.tensor([[0, 1, 2], [0, 1, 2]]) # 坐标(0, 0)、(1, 1)、(2, 2)
v = torch.tensor([1, 2, 3]) # 对应的值分别是1、2、3
a = torch.sparse_coo_tensor(i, v, (4, 4), dtype=torch.float32, device=dev) # 传入坐标、值、与张量大小
print(a)

# 将稀疏的张量转换为稠密张量
a = torch.sparse_coo_tensor(i, v, (4, 4), dtype=torch.float32, device=dev).to_dense()
print(a)