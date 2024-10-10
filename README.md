# FDU-AI-Homework

### author: TheEin
### collaborators: sbx，ChatGPT

### Tasks:
- Task1: 绘制损失函数的图像/理解损失函数的意义✅
  - 思考问题1:在训练集上损失得越少，模型就训练得越好吗？
  - 在训练集上损失得越少不一定越好，有可能过拟合而在测试集上表现不佳
- Task2: 加入正则化，提高优化器的性能和模型命中率
- Task3: 超参数调整，并分析影响


### Notes:
- 数据集CIFAR-10被置于项目之外，以便独立的管理
- 在本地运行时可以考虑更改：（令download=True）
```python
trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=transform)
```
- 关于损失函数图像的绘制，没有设置保存图像。并且非交互模式，需要及时关闭图像窗口