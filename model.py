# 导入必要的库
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torch.xpu import device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# 定义最佳超参数组合
BEST_PARAMS = {
    'dropout_prob': 0.35,
    'learning_rate': 0.002,
    'momentum': 0.94,
    'batch_size': 96,
    'num_epoch': 26,
    'weight_decay': 0.001
}


# def draw(values, title='Loss Function', xlabel='Epochs', ylabel='Loss'):
#     plt.figure(figsize=(10, 5))  # 设置图表大小
#     plt.plot(values, marker='o', linestyle='-', color='b')  # 绘制线图，添加点标记
#     plt.title(title)  # 添加标题
#     plt.xlabel(xlabel)  # 添加X轴标签
#     plt.ylabel(ylabel)  # 添加Y轴标签
#     plt.grid(True)  # 显示网格

#     # 设置X轴的刻度，使其每个单位长度为1
#     plt.xticks(range(len(values)))
#     plt.savefig("loss_function.png")  # 保存损失函数图像
#     plt.show()  # 显示图表


class Dropout_Net(nn.Module):
    def __init__(self, dp=0.4):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Dropout_Net, self).__init__()

        # 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层，y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=dp)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # reshape，‘-1’表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 模型的训练与测试
def train(trainloader, net, num_epochs, criterion, optimizer):
    loss_values = []
    scaler = torch.cuda.amp.GradScaler()  # 初始化 GradScaler
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # 使用自动混合精度
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()  # 缩放损失
            scaler.step(optimizer)  # 更新权重
            scaler.update()  # 更新缩放器

            running_loss += loss.item()

        average_loss = running_loss / len(trainloader)
        loss_values.append(average_loss)

# 在测试集上的表现
def predict(testloader, net):
    net.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 使用 .item() 获取标量值

    return 100 * correct / total


# 设置Dropout层
dropout_net = Dropout_Net().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(dropout_net.parameters(), lr=BEST_PARAMS['learning_rate'], momentum=BEST_PARAMS['momentum'],
                      weight_decay=0.001)  # 使用SGD（随z机梯度下降）优化/加入L2正则项
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 设置数据集路径
# dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')
dataset_path = './dataset'
# 设定训练和测试集

trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BEST_PARAMS['batch_size'], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BEST_PARAMS['batch_size'], shuffle=False, num_workers=4,pin_memory=True, persistent_workers=True)