import os
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim


# 定义用于分类的神经网络（与之前的代码相同）
class Dropout_Net(nn.Module):
	def __init__(self, dropout_prob=0.5):
		super(Dropout_Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 5)
		self.conv2 = nn.Conv2d(16, 32, 5)
		self.conv3 = nn.Conv2d(32, 64, 3)
		self.fc1 = nn.Linear(64, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 10)
		self.dropout = nn.Dropout(dropout_prob)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
		x = x.view(x.size()[0], -1)
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


# 定义训练函数
def train(trainloader, net, num_epochs, criterion, optimizer):
	correct = 0
	total = 0
	for epoch in range(num_epochs):
		for inputs, labels in trainloader:
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

	# 在测试集上的表现
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	return 100 * correct / total  # 返回准确率


if __name__ == '__main__':
	# 数据预处理和加载
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	batch_size = 4
	dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')
	trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

	# 超参数组合
	dropouts = [0.2, 0.4, 0.5]
	learning_rates = [0.001, 0.01]
	momentums = [0.8, 0.9]
	batch_sizes = [4, 8]
	num_epochs = [5, 6, 7]

	results = []

	for dropout, lr, momentum, b_size, n_e in product(dropouts, learning_rates, momentums, batch_sizes, num_epochs):
		print(f'Testing with dropout={dropout}, lr={lr}, momentum={momentum}, batch_size={b_size}, num_epochs={n_e}')
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=2)
		net = Dropout_Net(dropout_prob=dropout)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

		accuracy = train(trainloader, net, num_epochs=n_e, criterion=criterion, optimizer=optimizer)
		results.append((dropout, lr, momentum, b_size,n_e ,accuracy))

	# 排序并输出结果
	results.sort(key=lambda x: x[-1], reverse=True)
	for result in results:
		print(
			f'Dropout: {result[0]}, LR: {result[1]}, Momentum: {result[2]}, Batch Size: {result[3]},Num Epochs:{result[4]} ,Accuracy: {result[5]}%')
