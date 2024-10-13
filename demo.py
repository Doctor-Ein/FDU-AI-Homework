import time

from extreme_model import BEST_PARAMS, train, predict, criterion, trainloader, testloader, dropout_net, optimizer

# safe launch by using main as the entrance of this program
if __name__ == '__main__':
    print(dropout_net)
    # 训练模型
    print('Start Training')
    start_time = time.time()
    loss_values = train(trainloader, dropout_net, BEST_PARAMS['num_epoch'], criterion, optimizer)
    print('Finished Training')
    # 在测试集上预测/predict返回其命中率
    print('测试集中的准确率为: %d %%' % (predict(testloader, dropout_net)))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time:{execution_time:.2f} seconds.")
