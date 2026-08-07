import torch
import numpy as np
import re

# data
ff = open("housing.data").readlines()
data = []
for item in ff:
    out = re.sub(r"\s{2,}", " ", item).strip() # 将两个及以上的空格压缩为一个，同时去掉前后的空白字符例如\n等
    print(out)
    data.append(out.split(" ")) # 按空格划分列表，一行一行添加到data中
data = np.array(data).astype(float)
print(data.shape)

Y = data[:, -1] # 取最后一列房价作为输出标签
X = data[:, 0:-1] # 取前13列（除了最后一列）作为输入

# 划分出前496行数据作为训练集，后面的行作为测试集
X_train = X[0:496, ...]
Y_train = Y[0:496, ...]
X_test = X[496:, ...]
Y_test = Y[496:, ...]

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# net
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        # 参数为输入特征数、输出标签数
        # 线性网络
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_output)

    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out

net = Net(13, 1)

# loss
loss_func = torch.nn.MSELoss() # 损失函数为均方损失函数

# optimiter
# 优化器，采用Adam优化，传入参数以及学习率
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# training
for i in range(10000):
    x_data = torch.tensor(X_train, dtype=torch.float32)
    y_data = torch.tensor(Y_train, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred) # pred1前向计算得出的结果是一张二维表[496, 1]，而y_data是一维的，所以去掉1这个维度
    loss = loss_func(pred, y_data) * 0.001 # 计算前向传播得出的值与标签的差距

    optimizer.zero_grad() # 梯度清零，不然会累加
    loss.backward() # 反向传播
    optimizer.step() # 根据梯度去更新参数

    print("ite:{}, loss_train:{}".format(i, loss))
    print(pred[0:10])
    print(y_data[0:10])

    #test
    x_data = torch.tensor(X_test, dtype=torch.float32)
    y_data = torch.tensor(Y_test, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, y_data) * 0.001
    print("ite:{}, loss_test:{}".format(i, loss_test))

# torch.save(net, "model/model.pkl") # 保存整个模型，PyTorch2.6之后把该方法改得更安全了,所以需要更严格的加载
# torch.load("") # 加载时直接torch.load("")
torch.save(net.state_dict(), "model/params.pth") # 保存参数
# net.load_state_dict("") # 加载时需要定义模型后恢复参数