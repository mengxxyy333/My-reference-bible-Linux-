import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN
import cv2

# data
# 参数依次：数据集存放位置、是否为训练集、转换为tensor，如果数据集没有需要下载
test_data = dataset.MNIST(root="mnist", train=False, transform=transforms.ToTensor(), download=False)

# 指定训练集、设置batch_size、打乱数据排列
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=64, shuffle=True)

cnn = CNN()
state_dict = torch.load("model/mnist_model.pth")
cnn.load_state_dict(state_dict)
cnn = cnn.cuda()

# test/eval
loss_test = 0
accuracy = 0
for i, (images, labels) in enumerate(test_loader):
    images = images.cuda()
    labels = labels.cuda()
    outputs = cnn(images)
    _, pred = outputs.max(1)
    accuracy += (pred == labels).sum().item()

    # 拿到图片数据,可视化部分
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()
    # batchsize * 1 * 28 * 28
    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_label = labels[idx]
        im_pred = pred[idx]
        im_data = im_data.transpose(1, 2, 0)

        print("label", im_label)
        print("pred", im_pred)
        cv2.imshow("imdata", im_data)
        cv2.waitKey(0)

accuracy = accuracy / len(test_data)
print(accuracy)