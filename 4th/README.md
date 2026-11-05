# 实验四：MLP-多层感知机
### 202410315012_张连艺

## 一、实验目的
1. 掌握 PyTorch 基础框架使用，理解数据集加载、数据变换、数据加载器DataLoader工作流程。
2. 理解多层感知机（MLP）神经网络结构原理，掌握全连接层、激活函数、损失函数、优化器的使用。
3. 使用 MLP 完成 MNIST 手写数字数据集的分类任务，训练模型并评估测试集识别准确率。
4. 理解神经网络训练、验证循环流程，掌握训练模式`model.train()`与评估模式`model.eval()`的区别，熟悉梯度清零、反向传播、参数更新完整训练流程。

## 二、实验内容
### 1.导入必要库
- `torch`: PyTorch深度学习框架的核心库，提供张量操作和基本功能
- `torch.nn`: PyTorch的神经网络模块，包含各种层、损失函数和模型定义类
- `torch.nn.functional`: 包含神经网络相关的函数式接口，如激活函数、损失函数等
- `torch.optim`: 优化算法模块，提供SGD、Adam等优化器
- `numpy`: 数值计算库，用于数组操作和数学运算
- `matplotlib.pyplot`: 数据可视化库，用于绘制图表和图像显示
- `torchvision`: PyTorch的计算机视觉库，提供：
- `datasets`: 常用数据集加载（如MNIST等）
- `transforms`: 数据预处理和增强工具

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
```

### 2.设置超参数、设备、加载数据集与 DataLoader
- `batch_size = 2048`:设置批量大小参数，每次训练时同时处理2048个样本
- `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`:有GPU → 选择GPU加速，无GPU → 选择CPU计算
-  `train_loader`:加载 MNIST手写数字数据集训练集
- `test_loader`:加载 MNIST手写数字数据集测试集
- `transform`:将PIL图像或numpy数组转换为PyTorch张量

```python
batch_size = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
```python
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=batch_size, shuffle=True)

print("训练数据形状:", train_loader.dataset.data.shape)
print("测试数据形式:", test_loader.dataset.data.shape)
```

```
输出：
    训练数据形状: torch.Size([60000, 28, 28])
    测试数据形式: torch.Size([10000, 28, 28])
```

### 3.定义 MLP 网络模型
继承`nn.Module`（神经网络模块基类）搭建 MLP 模型。
1. 在`__init__()`初始化函数中定义两层全连接层：
- `self.l1 = nn.Linear(784, 128)`：输入 784 维（28×28 图像展平），输出 128 维隐藏层；
- `self.l2 = nn.Linear(128, 10)`：输入 128 维隐藏层，输出 10 维，对应 0‑9 共 10 个数字类别。

2. `forward()`实现前向传播流程：
- 经过第一层线性变换，再通过`F.relu()`激活函数引入非线性；
- 第二层线性变换输出分类得分。

3. 实例化模型与优化器：
- `model = MLP().to(device)`实例化模型，并迁移至 GPU/CPU 设备；
- `optimizer = optim.SGD(model.parameters(), lr=0.1)`选用 SGD 随机梯度下降，设置学习率 0.1 更新网络参数。
 
```python
class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        # 第一层全连接：784输入 →128输出
        self.l1 = nn.Linear(784, 128)
        # 第二层全连接：128输入 →10输出（10个数字类别）
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        a1 = self.l1(x)
        x1 = F.relu(a1)   # ReLU激活函数
        a2 = self.l2(x1)
        return a2

# 实例化模型并迁移到对应设备(GPU/CPU)
model = mlp().to(device)
# SGD优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)
print(model)

```
```python
输出：
mlp(
  (l1): Linear(in_features=784, out_features=128, bias=True)
  (l2): Linear(in_features=128, out_features=10, bias=True)
)
```

### 4.模型迭代训练和测试评估循环
训练阶段：
- `epochs = 10`: 定义整体迭代轮数，完整遍历训练集 10 次
- `for epoch in range(epochs):`: 外层循环，控制数据集整体迭代次数，重复训练数据集 10 轮
- `model.train()`: 将模型切换为**训练模式**，启用与训练相关层的特性
- `for batch_idx, (x, y) in enumerate(train_loader):`：内层批次循环，按 batch 分批读取训练集数据，`batch_idx`为批次编号，`x`为图像张量，`y`为对应标签
- `x, y = x.view(x.shape[0], -1).to(device), y.to(device)`: 数据预处理，把二维图像展平为一维特征向量，同时将图像、标签张量迁移至 CPU/GPU 同一设备
- `optimizer.zero_grad()`: 每个批次训练前清零梯度，避免不同批次梯度累积造成参数更新错误
- `output = model(x)`：前向传播，将特征送入网络，输出各个类别的预测得分
- `loss = F.cross_entropy(output, y)`：使用内置 softmax 的交叉熵损失函数，计算该批次预测结果与真实标签之间的损失
- `loss.backward()`：反向传播，根据损失自动求解网络各参数的梯度
- `optimizer.step()`：优化器依据梯度更新网络权重参数，完成单批次训练，继续循环下一个批次，直到本 epoch 所有批次遍历完毕

```python
epochs = 30
for epoch in range(epochs):
    # ==========训练阶段==========
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        # 将28×28图片展平为784维向量，并搬运到设备
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        optimizer.zero_grad()       # 梯度清零
        output = model(x)           # 前向传播
        loss = F.cross_entropy(output, y) # 计算交叉熵损失
        loss.backward()             # 反向传播求梯度
        optimizer.step()            # 更新参数

```

测试评估阶段:
- `model.eval()`: 将模型切换为**评估模式**，关闭训练专属逻辑，保证测试推理环境
- `correct = 0`、`test_loss = 0`: 初始化变量，分别用于统计预测正确样本总数、累计测试集损失
- `with torch.no_grad():`：关闭梯度计算，测试阶段不需要反向传播，节约内存、提升推理速度
- `for batch_idx, (x, y) in enumerate(test_loader):`：分批次读取测试集数据，对全部测试样本做推理
- `x, y = x.view(x.shape[0], -1).to(device), y.to(device)`：测试集图像同样展平为一维向量，并迁移到对应运算设备
- `output = model(x)`：前向传播，得到测试样本的分类预测得分
- `test_loss += F.cross_entropy(output, y)`：累加每一批次的损失，统计整体测试损失
- `pred = output.max(1, keepdim=True)[1]`：取得分最大下标作为样本的预测类别
- `correct += pred.eq(y.view_as(pred)).sum().item()`：对比预测值和真实标签，统计预测正确样本数量
- `test_loss = test_loss / (batch_idx+1)`：求平均测试损失
- `acc = correct / len(test_loader.dataset)`：计算测试集整体分类准确率
- `print(f'epoch:{epoch:2d}, loss:{test_loss:.4f}, acc:{acc:.4f}')`：打印本轮 epoch 的测试损失与准确率，完成一轮训练‑评估流程，进入下一轮 epoch 循环。
  
```python
    # ==========测试评估阶段==========
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad(): # 关闭梯度计算
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.view(x.shape[0], -1).to(device), y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output, y)
            pred = output.max(1, keepdim=True)[1] # 获取预测类别
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss = test_loss / (batch_idx+1)
    acc = correct / len(test_loader.dataset)
    print(f'epoch:{epoch:2d}, loss:{test_loss:.4f}, acc:{acc:.4f}')
```

```python
输出结果：
epoch: 0, loss:0.5934, acc:0.8624
epoch: 1, loss:0.5805, acc:0.8639
epoch: 2, loss:0.5696, acc:0.8662
epoch: 3, loss:0.5585, acc:0.8676
epoch: 4, loss:0.5485, acc:0.8687
epoch: 5, loss:0.5380, acc:0.8703
epoch: 6, loss:0.5297, acc:0.8709
epoch: 7, loss:0.5214, acc:0.8726
epoch: 8, loss:0.5140, acc:0.8737
epoch: 9, loss:0.5064, acc:0.8754
epoch:10, loss:0.4988, acc:0.8773
epoch:11, loss:0.4922, acc:0.8780
epoch:12, loss:0.4861, acc:0.8789
epoch:13, loss:0.4803, acc:0.8798
epoch:14, loss:0.4745, acc:0.8808
epoch:15, loss:0.4686, acc:0.8823
epoch:16, loss:0.4636, acc:0.8828
epoch:17, loss:0.4585, acc:0.8844
epoch:18, loss:0.4544, acc:0.8854
epoch:19, loss:0.4506, acc:0.8857
epoch:20, loss:0.4455, acc:0.8865
epoch:21, loss:0.4413, acc:0.8871
epoch:22, loss:0.4375, acc:0.8881
epoch:23, loss:0.4333, acc:0.8882
epoch:24, loss:0.4306, acc:0.8892
epoch:25, loss:0.4267, acc:0.8897
epoch:26, loss:0.4240, acc:0.8899
epoch:27, loss:0.4202, acc:0.8905
epoch:28, loss:0.4179, acc:0.8912
epoch:29, loss:0.4140, acc:0.8922
```

## 三、实验结果与分析
### 1.关键知识点分析
1. `view(x.shape[0],‑1)`：把`[batch,1,28,28]`的四维张量展平成`[batch,784]`二维，全连接层只接受一维特征输入。
2. `model.train()`与`model.eval()`：`train()`启用训练模式；`eval()`评估模式，禁用 dropout、停止 batchnorm 更新统计量，测试阶段必须开启。
3. `optimizer.zero_grad()`：每一轮 batch 训练前必须清零梯度，否则梯度会累积，参数更新错误。
4. `with torch.no_grad()`：测试阶段不需要反向传播，关闭梯度计算，节省显存、加速推理。
5. CrossEntropyLoss：用于多分类，输入不需要手动做 softmax，函数内部自动完成。

### 2.误差分析
模型最终准确率没有达到 90% 以上，原因：
1. 网络结构简单，仅两层全连接，特征提取能力有限，没有卷积层提取局部空间特征。
2. 超参数限制：学习率 0.01、迭代 30 轮、batch_size=2048，可以调整学习率、增加隐层神经元、增加训练轮数进一步提升准确率。

## 四、实验总结与心得
本次实验完成基于 PyTorch 的 MLP 多层感知机对 MNIST 手写数字的分类任务。掌握了数据集下载加载、`DataLoader`批量读取、网络模型定义、训练‑测试循环完整流程。

MNIST 原始 28×28 灰度图像，需要展平为 784 维向量送入全连接网络；ReLU 激活函数为网络引入非线性；SGD 优化器配合交叉熵损失完成权重更新。训练阶段开启`model.train()`，测试阶段切换`model.eval()`，使用`no_grad()`关闭梯度计算。

两层 MLP 可以完成手写数字识别任务，随着训练轮数增加损失降低、准确率上升。但单纯全连接网络对图像空间特征提取能力有限，识别效果还有提升空间，后续可以引入卷积神经网络 CNN 进一步提高手写数字识别准确率。
