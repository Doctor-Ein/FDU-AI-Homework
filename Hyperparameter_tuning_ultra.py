# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# 定义最佳超参数组合
dropout = 0
lr = 0.01
momentum = 0.9
batch_size = 64
num_epochs = 16
weight_decay = 0.005


class Dropout_Net(nn.Module):
	def __init__(self, dp=0.2):
		super(Dropout_Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.bn1 = nn.BatchNorm2d(6)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.bn2 = nn.BatchNorm2d(16)
		self.conv3 = nn.Conv2d(16, 32, 5)
		self.bn3 = nn.BatchNorm2d(32)
		self.fc1 = nn.Linear(32 * 1 * 1, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
		self.dropout = nn.Dropout(p=dp)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2, 2))
		x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
		x = F.relu(self.bn3(self.conv3(x)))  # 去掉这一层的池化
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

			running_loss += loss.item()  # 收集损失值

		average_loss = running_loss / len(trainloader)
		loss_values.append(average_loss)

	return loss_values  # 返回损失值以便进一步处理或绘图


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


# 初始化数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset_path = './dataset'  # 设置数据集路径
trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
criterion = nn.CrossEntropyLoss()
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                         persistent_workers=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                        persistent_workers=True)
dropout_net = Dropout_Net(dropout).to(device)  # 假设你的模型类名为 Dropout_Net
optimizer = optim.SGD(dropout_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
# 训练和评估
train(trainloader, dropout_net, num_epochs, criterion, optimizer)
accuracy = predict(testloader, dropout_net)
print("%.1f %", accuracy)
