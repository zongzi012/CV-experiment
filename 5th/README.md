# 实验五：CNN-卷积神经网络
### 202410315012-张连艺

## 一、实验目的
1. 掌握 PyTorch 搭建卷积神经网络的基本流程，理解`nn.Module`网络类的编写规范，学会定义卷积层、池化层、全连接层。
2. 理解 LeNet‑5 网络结构，熟悉卷积、平均池化、Sigmoid 激活、全连接层在手写数字识别任务中的作用。
3. 掌握数据集加载、DataLoader 批量读取数据、GPU/CPU 设备迁移、损失函数、优化器、反向传播训练流程。
4. 掌握模型测试评估方法，使用测试集计算损失与分类准确率，观察迭代过程中损失、准确率变化。
5. 能够提取卷积层输出的特征图，可视化卷积提取出来的图像特征。

## 二、实验内容
### 1.导入库
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
```

### 2.设置设备和参数
1. `batch_size = 512`:设置批量大小参数，每次训练时同时处理512个样本
2. `device = torch.device("cpu")`:选择CPU计算

```python
batch_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 3.加载MNIST数据集
1. 训练数据加载器 `train_loader`:加载 MNIST手写数字数据集训练集
2. 测试数据加载器 `test_loader`:加载 MNIST手写数字数据集测试集
3. 数据预处理`transform`:将图像转换为张量

```python
trainloader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True,
              transform=transforms.Compose([transforms.ToTensor()])), batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, download=True,
              transform=transforms.Compose([transforms.ToTensor()])), batch_size=batch_size, shuffle=True)

print("训练数据形状:", trainloader.dataset.data.shape)
print("测试数据形式:", testloader.dataset.data.shape)
```

```
输出：
    训练数据形状: torch.Size([60000, 28, 28])
    测试数据形式: torch.Size([10000, 28, 28])
```

### 4.创建模型
1. `class Net(nn.Module)`：所有 pytorch 网络都要继承`nn.Module`
2. `Conv2d`卷积层：
   `conv1`：输入是单通道灰度图，输出 6 个特征图，padding=2 让卷积后图像大小不变。
   `conv2`：6 通道输入，输出 16 个特征图。
3. `nn.Linear`全连接层：把卷积提取的特征映射成向量。
4. `forward()`：数据从输入到输出的计算流程，**不要在 init 里面写激活、池化**。
   `F.sigmoid()`激活函数；`F.avg_pool2d`平均池化，降低图像尺寸，减少参数量。
   `torch.flatten(x,1)`：把除 batch 之外的维度全部摊平，给全连接层使用。
5. `model.to(device)`把网络移到 GPU/CPU；`Adam`优化器，学习率 0.01；epochs=30 训练轮数。

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2) # 输入通道1，输出通道6，卷积核5x5
        self.conv2 = nn.Conv2d(6, 16, 5) # 输入通道6，输出通道16，卷积核5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 全连接层，输入16*5*5，输出120
        self.fc2 = nn.Linear(120, 84) # 全连接层，输入120，输出84
        self.clf = nn.Linear(84, 10) # 分类层，输入84，输出10

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        # 激活函数sigmoid()
        x = F.sigmoid(x)
        # 平均池化层，kernel=2x2，步长2
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        # conv2
        x = self.conv2(x)
        # 激活函数sigmoid()
        x = F.sigmoid(x)
        # 平均池化层，2x2，步长2
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        # 展开，从第1维开始展开
        x = x.view(x.size(0), -1)
        
        # 全连接层1
        x = self.fc1(x)
        # 激活函数sigmoid()
        x = F.sigmoid(x)
        # 全连接层2
        x = self.fc2(x)
        # 激活函数sigmoid()
        x = F.sigmoid(x)
        
        # 分类层
        x = self.clf(x)
        return x

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
model
```

```
输出：
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (clf): Linear(in_features=84, out_features=10, bias=True)
)
```

### 5.训练与测试
1. 外层循环`epoch`，一轮代表完整跑一遍全部训练集。
2. 内层遍历训练集每个 batch：
   - `x,y = x.to(device),y.to(device)`数据迁移到 GPU/CPU；
   - `optimizer.zero_grad()`每批次必须清空历史梯度，防止梯度累积；
   - `loss.backward()`反向传播计算参数梯度；`optimizer.step()`更新网络权重。
3. `with torch.no_grad()`测试阶段关闭梯度计算，节省显存，不更新参数。
4. `out.max(dim=1)[1]`拿到预测的数字标签；统计预测正确样本，计算测试集准确率。

```python
# 训练
epochs = 30
accs, losses = [], []
for epoch in range(epochs):
    model.train()
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
# 测试循环
    correct = 0
    testloss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            testloss += F.cross_entropy(out, y).item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(y).sum().item()

    acc = correct / len(testloader.dataset)
    testloss /= (batch_idx + 1)
    accs.append(acc)
    losses.append(testloss)
    print('epoch:{}, loss:{:.4f}, acc:{:.4f}'.format(epoch, testloss, acc))
```

```
训练结果：
epoch:0, loss:2.3013, acc:0.1135
epoch:1, loss:1.4100, acc:0.4686
epoch:2, loss:0.2615, acc:0.9174
epoch:3, loss:0.1532, acc:0.9509
epoch:4, loss:0.0905, acc:0.9713
epoch:5, loss:0.0772, acc:0.9765
epoch:6, loss:0.0711, acc:0.9776
epoch:7, loss:0.0730, acc:0.9778
epoch:8, loss:0.0514, acc:0.9839
epoch:9, loss:0.0469, acc:0.9861
epoch:10, loss:0.0410, acc:0.9867
epoch:11, loss:0.0453, acc:0.9859
epoch:12, loss:0.0439, acc:0.9856
epoch:13, loss:0.0405, acc:0.9874
epoch:14, loss:0.0400, acc:0.9880
epoch:15, loss:0.0505, acc:0.9846
epoch:16, loss:0.0416, acc:0.9873
epoch:17, loss:0.0464, acc:0.9883
epoch:18, loss:0.0516, acc:0.9853
epoch:19, loss:0.0390, acc:0.9889
epoch:20, loss:0.0405, acc:0.9881
epoch:21, loss:0.0419, acc:0.9887
epoch:22, loss:0.0407, acc:0.9888
epoch:23, loss:0.0389, acc:0.9902
epoch:24, loss:0.0423, acc:0.9890
epoch:25, loss:0.0472, acc:0.9882
epoch:26, loss:0.0418, acc:0.9886
epoch:27, loss:0.0472, acc:0.9878
epoch:28, loss:0.0385, acc:0.9900
epoch:29, loss:0.0477, acc:0.9878
```

可视化分析：
```python
# 可视化
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(losses, label='Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()

plt.subplot(1,2,2)
plt.plot(accs, label='Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()

plt.show()
```
运行结果：

<img width="541" height="250" alt="image" src="https://github.com/user-attachments/assets/c9c6ddf4-baef-4548-a5b5-396718ae9a7d" />

  
### 6.可视化每层卷积的效果
1. 跳过池化层，直接取出卷积层经过 sigmoid 后的输出特征图。
2. `.detach().cpu().numpy()`：把张量从 GPU 拿回到 CPU，转为 numpy 数组，用于绘图。
3. `n=5`选择 5 张样本；`sum(0)`对通道维度求和，把多通道特征图合并成单张灰度图方便观察。
4. 绘图分为三行：原始图片、第一层卷积特征、第二层卷积特征，直观看到卷积网络提取出来的图像纹理轮廓。

```
feature1 = F.sigmoid(model.conv1(x))
# feature1 = F.avg_pool2d(feature1,kernel_size=2,stride=2)
feature2 = F.sigmoid(model.conv2(feature1))
# feature2 = F.avg_pool2d(feature2,kernel_size=2,stride=2)

n = 5
img = x.detach().cpu().numpy()[:n]
feature_map1 = feature1.detach().cpu().numpy()[:n]
feature_map2 = feature2.detach().cpu().numpy()[:n]

fig, ax = plt.subplots(3, n, figsize=(10,10))
for i in range(n):
    ax[0,i].imshow(img[i].sum(0),cmap='gray')
    # ax[0,i].axis('off')
    ax[1,i].imshow(feature_map1[i].sum(0),cmap='gray')
    # ax[1,i].axis('off')
    ax[2,i].imshow(feature_map2[i].sum(0),cmap='gray')
    # ax[1,i].axis('off')
plt.show()
```
运行结果：

原始输入图像：

<img width="550" height="125" alt="image" src="https://github.com/user-attachments/assets/94b6f3d1-9a1b-4317-b594-28ff22fe1ebf" />

conv1 第一层卷积输出特征图：
<img width="544" height="122" alt="image" src="https://github.com/user-attachments/assets/e1fc1996-58af-4fdf-b0a1-0c327b49fd69" />

conv2 第二层卷积输出特征图：
<img width="551" height="124" alt="image" src="https://github.com/user-attachments/assets/ed8e992e-5ae0-4f44-ab97-59e0d0c3e775" />



## 三、实验结果与分析
本次实验使用LeNet网络对MNIST数字集进行了训练、测试和可视化分析的过程。
### 1.通过训练日志和可视化训练曲线分析：
从最后输出的训练日志结果分析得出，模型最终的测试准确率稳定在98%以上，说明网络已经能够比较充分学习数据，对于分类来说非常有效了。<br>
从可视化损失曲线和准确率曲线也可以看出，模型在训练过程中没有过拟合现象，也没有欠拟合，整体表现非常好。<br>

### 2.通过混淆矩阵分析：
我还自己画了一张训练预测结果的混淆矩阵，从混淆矩阵可以看到大多数数字的预测都集中在对角线上，说明模型对准确较高的。少量的错误通常出现于形状比较相似的数字之间，例如4和9、5和3、7和9等，这几对数字本身在书写上就容易混淆，所以预测会大概率出现偏差。<br>

### 3.通过网络层结果图分析：
最后我制作了可视化卷积层与池化层的特征图，从中更直观看到网络提取特征的过程：浅层主要关注的是数字的边缘笔画等基础结构；而随着网络的加深，特征图内容变得更加抽象，如数字的弯曲形交叉结构等。池化层则让特征变得更加简洁，同时减少噪声，让网络更容易从中提取关键信息。<br>
从可视化图像看到，第一层卷积层提取图像的基础低级特征，如边缘笔画等；第一次池化层的图像的尺寸被缩小为原来的一半，但主要结构仍被保留，噪声被有效抑制。在第二层卷积层中，特征图比第一层更加抽象，卷积核开始关注更高层次的组合结构，例如数字中的特定弯曲交叉形状；第二次池化进一步压缩空间维度。<br>
虽然卷积和池化操作到最后的特征图对于人类来说无法分辨和解释，但是对于计算机和CNN算法来说，这样的卷积加池化使得网络能够专注于更加关键的判别特征。<br>
总体来说，本次实验的模型不仅在指标上有不错的表现，通过可视化也能明显看到CNN从低层到高层逐步提取特征的过程。

## 四、实验总结与心得
