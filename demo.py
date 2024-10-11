from torch import optim

from model import BEST_PARAMS, Dropout_Net, train, predict, criterion, trainloader, testloader, draw

# safe launch by using main as the entrance of this program
if __name__ == '__main__':
	# 设置Dropout层
	dropout_net = Dropout_Net()
	print(dropout_net)
	optimizer = optim.SGD(dropout_net.parameters(), lr=BEST_PARAMS['learning_rate'], momentum=BEST_PARAMS['momentum'],
	                      weight_decay=0.001)  # 使用SGD（随z机梯度下降）优化/加入L2正则项
	# 训练模型
	draw(train(trainloader, dropout_net, BEST_PARAMS['num_epoch'], criterion, optimizer, save_path='TrainDataCollection'))
	print('Finished Training')
	# 在测试集上预测/predict返回其命中率
	print('测试集中的准确率为: %d %%' % (predict(testloader, dropout_net)))
