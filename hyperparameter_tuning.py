from itertools import product
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn

from ultra_model import device, Dropout_Net, train, predict

if __name__ == '__main__':
    # 超参数组合
    dropouts = [0.2, 0.3, 0.4, 0.5]
    learning_rates = [0.001, 0.01]
    momentums = [0.9, 0.99]
    batch_sizes = [32, 64, 128, 196]
    num_epochs = [4 ,8, 16,24]
    weight_decays = [0, 0.001, 0.005, 0.01]
    results = []

    i = 0
    with open("OutputTemp.txt", "w") as file:
        file.write("Hyperparameter_tuning: \n")
    # 数据预处理
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_path = './dataset' # 设置数据集路径
    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
    criterion = nn.CrossEntropyLoss()
    for dropout, lr, momentum, b_size, num_epoch, weight_decay in product(dropouts, learning_rates, momentums, batch_sizes, num_epochs, weight_decays):
        print(f'''Testing with dropout={dropout}, lr={lr}, momentum={momentum}, batch_size={b_size}, num_epochs={num_epoch}, weight_decay={weight_decay}''')
        # 超参数相关：
        trainloader = DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        testloader = DataLoader(testset, batch_size=b_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        dropout_net = Dropout_Net(dropout).to(device)  # 假设你的模型类名为 Dropout_Net
        optimizer = optim.SGD(dropout_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        # 训练和评估
        train(trainloader, dropout_net, num_epoch, criterion, optimizer)
        accuracy = predict(testloader, dropout_net)
        results.append((dropout, lr, momentum, b_size, num_epoch, weight_decay, accuracy))
        result = results[-1]
        i += 1
        with open("OutputTemp.txt", "a") as file:
            file.write("num_params:" + str(i) + f' ,Dropout: {result[0]}, LR: {result[1]}, Momentum: {result[2]}, Batch Size: {result[3]}, Num Epochs: {result[4]}, Weight Decay: {result[5]}, Accuracy: {result[6]:.1f}%\n')
        # sleep(60)  # 请让电脑休息一分钟，也请记得监控计算机资源占情况，确保程序和设备安全
    # 排序并输出结果
    results.sort(key=lambda x: x[-1], reverse=True)
    for result in results:
        with open("OutputFinal.txt", "w") as file:
            file.write(
                f'Dropout: {result[0]}, LR: {result[1]}, Momentum: {result[2]}, Batch Size: {result[3]}, Num Epochs: {result[4]}, Weight Decay: {result[5]}, Accuracy: {result[6]:.1f}%\n')
# TODO：最佳超参数组合的自动写入
