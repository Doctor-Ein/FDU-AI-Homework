# 导入必要的库
import os  # 用于文件路径处理

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim


def draw(values, title='Loss Function', xlabel='Epochs', ylabel='Loss'):
	plt.figure(figsize=(10, 5))  # 设置图表大小
	plt.plot(values, marker='o', linestyle='-', color='b')  # 绘制线图，添加点标记
	plt.title(title)  # 添加标题
	plt.xlabel(xlabel)  # 添加X轴标签
	plt.ylabel(ylabel)  # 添加Y轴标签
	plt.grid(True)  # 显示网格

	# 设置X轴的刻度，使其每个单位长度为1
	plt.xticks(range(len(values)))
	plt.show()  # 显示图表


# plt.savefig("loss_function.png") 保存损失函数图像


# 定义用于分类的神经网络
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.conv3 = nn.Conv2d(16, 32, 3)  # 新增卷积层
		self.fc1 = nn.Linear(32 * 3 * 3, 120)  # 注意这里的尺寸需要根据新层调整
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = F.relu(self.conv3(x))  # 移除最后一个池化层
		x = x.view(x.size()[0], -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class Dropout_Net(nn.Module):
	def __init__(self):
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
		self.dropout = nn.Dropout(p=0.4)  # 概率为0.4的Dropout层

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
def train(trainloader, net, num_epochs, criterion, optimizer, save_path):
	loss_values = []  # 用于存储每个epoch的平均损失值
	for epoch in range(num_epochs):
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# 取出数据
			inputs, labels = data

			# 梯度清零
			optimizer.zero_grad()

			# 计算损失函数
			outputs = net(inputs)
			loss = criterion(outputs, labels)

			# 反向传播
			loss.backward()
			optimizer.step()
			running_loss += loss.item()  # 累加每个批次的损失

		average_loss = running_loss / len(trainloader)
		loss_values.append(average_loss)  # 将平均损失添加到列表

	# 将损失值列表转换为张量并绘制
	draw(torch.tensor(loss_values))


# 在测试集上的表现
def predict(testloader, net):
	correct = 0  # 预测正确的图片数
	total = 0  # 总共的图片数

	with torch.no_grad():  # 正向传播时不计算梯度
		for data in testloader:
			# 1. 取出数据
			images, labels = data
			# 2. 正向传播，得到输出结果
			outputs = net(images)
			# 3. 从输出中得到模型预测
			_, predicted = torch.max(outputs, 1)
			# 4. 计算性能指标
			total += labels.size(0)
			correct += (predicted == labels).sum()

	print('测试集中的准确率为: %d %%' % (100 * correct / total))


# safe launch by using main as the entrance of this program
if __name__ == '__main__':
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	batch_size = 4

	# 设置数据集路径
	dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')

	# 设定训练和测试集
	trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

	# show = ToPILImage()  #可视化数据图像的工具，不过没有用哩

	# # 设置基本的神经网络架构
	# net = Net()
	# print(net)  # 展示具体卷积网络的设置架构

	# 设置Dropout层
	dropout_net = Dropout_Net()
	print(dropout_net)

	# 模型的训练
	criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
	optimizer = optim.SGD(dropout_net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)  # 使用SGD（随z机梯度下降）优化/加入L2正则项
	num_epochs = 5  # 训练 5 个 epoch

	train(trainloader, dropout_net, num_epochs, criterion, optimizer, save_path='TrainDataCollection')
	print('Finished Training')

	# 在测试集上预测
	predict(testloader, dropout_net)
