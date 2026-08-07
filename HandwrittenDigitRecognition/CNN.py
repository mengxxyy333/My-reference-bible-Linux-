import torch

# net
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 序列方式定义网络
        # 卷积层
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            torch.nn.BatchNorm2d(64), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        # 线性层
        # 14*14是因为上面有一个pool层，会将图片尺寸减半
        # 输出10是因为需要0-9十个数字的概率
        self.fc = torch.nn.Linear(7 * 7 * 64, 10)
    
    def forward(self, x):
        out = self.conv(x) # 这里给出的结果仍然是二维
        out = out.view(out.size()[0], -1) # 所以要拉直成一维，再送入线性层
        self.dropout = torch.nn.Dropout(0.5)
        out = self.fc(out)
        return out