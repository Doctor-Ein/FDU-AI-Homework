# FDU-AI-Homework

### author: TheEin
### collaborators: sbx，ChatGPT

# How to run this program
- demo.py 是演示效果的脚本，运行它就足以应付作业了
- hyperparameter_tuning.py是自动交叉对比法得到最优超参数的脚本，首先运行它（可能会跑得有点久）

### Tasks:
- Task1: 绘制损失函数的图像/理解损失函数的意义✅
  - 思考问题1:在训练集上损失得越少，模型就训练得越好吗？
  - 在训练集上损失得越少不一定越好，有可能过拟合而在测试集上表现不佳
- Task2: 加入正则化，提高优化器的性能和模型命中率 ✅
  - 虽然但是正则化的参数设置也许会反向影响模型的命中率
- Task3: 超参数调整，并分析影响✅
  - 自动超参调整脚本
  - 影响拟合程度（即泛化能力）


### Notes:
- 数据集CIFAR-10被置于项目之外，以便独立的管理
- 在本地初次运行时可以考虑更改：（令download=True）以便保证数据集能够被下载
```python
trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=transform)
```
- 关于损失函数图像的绘制，没有设置保存图像。并且非交互模式，需要及时关闭图像窗口
