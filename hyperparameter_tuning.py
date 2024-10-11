from itertools import product

from torch import optim

from model import Dropout_Net, train, predict, criterion, trainloader, testloader

if __name__ == '__main__':
	# 超参数组合
	dropouts = [0.2, 0.3, 0.4, 0.5]
	learning_rates = [0.001, 0.01]
	momentums = [0.8, 0.9, 0.99]
	batch_sizes = [4, 8]
	num_epochs = [5, 6, 7]
	results = []

	for dropout, lr, momentum, b_size, num_epoch in product(dropouts, learning_rates, momentums, batch_sizes,
	                                                        num_epochs):
		print(
			f'''Testing with dropout={dropout}, lr={lr}, momentum={momentum}, batch_size={b_size}, num_epochs={num_epoch}''')
		dropout_net = Dropout_Net()
		optimizer = optim.SGD(dropout_net.parameters(), lr=lr, momentum=momentum, weight_decay=0.001)
		train(trainloader, dropout_net, num_epoch, criterion, optimizer, save_path='TrainDataCollection')
		accuracy = predict(testloader, dropout_net)
		results.append((dropout, lr, momentum, b_size, num_epoch, accuracy))
	# TODO：实际跑程序时需要注意sleep休息，同时记得监控计算机相关硬件占用信息，确保程序和设备安全

	# 排序并输出结果
	results.sort(key=lambda x: x[-1], reverse=True)
	for result in results:
		print(
			f'Dropout: {result[0]}, LR: {result[1]}, Momentum: {result[2]}, Batch Size: {result[3]},Num Epochs:{result[4]} ,Accuracy: {result[5]}%')
# TODO：最佳超参数组合的自动写入
