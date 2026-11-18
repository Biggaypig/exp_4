# 实验四 全连接神经网络

## 实验目的

实现多层感知机MLP在MNIST手写数字数据集上进行图像分类任务

## 实验内容

1. 运行代码实现图像分类
2. 分析代码运行结果并总结

## 实验结果与总结

1. 导入所需库


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
```

torch:PyTorch深度学习框架核心  
torch.nn: 神经网络模块  
torch.nn.functional: 函数式接口  
torch.optim: 优化算法  
numpy: 数值计算  
matplotlib.pyplot: 数据可视化  
torchvision: 计算机视觉相关工具,包含datasets(常用数据集)和transforms(数据预处理变换)  

2. 设置批量大小并指定计算设备


```python
batch_size = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

2048批量大小较大，有助于稳定训练，虽然会占用更多内存，但训练速度更快，梯度估计更准确，但也可能会降低泛化能力  
这里计算设备优先选择GPU，不支持再使用CPU计算

3. 加载MNIST数据集，并创建数据加载器


```python
train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),])),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(),])),
                                          batch_size=batch_size, shuffle=True)
```


```python
train_loader.dataset.data.shape
```




    torch.Size([60000, 28, 28])



MNIST数据集：手写数字数据集，包含0-9共10个类别，60,000个训练样本+10,000个测试样本，每张图片大小为28×28像素
数据变换：transforms.ToTensor()——将PIL图像或numpy数组转换为PyTorch张量，自动将像素值从[0,255]缩放到[0,1]，将图像维度从(H,W,C)调整为(C,H,W)  
DataLoader参数：  
batch_size: 每个批次的样本数  
shuffle=True: 训练时打乱数据顺序，防止模型记忆顺序  
这里将测试集也打乱，但通常测试集不需要shuffle

4. 定义MLP模型，包含两个全连接层（784->128->10），使用ReLU激活函数  


```python
class mlp(nn.Module):
    def __init__(self):
        super (mlp,self).__init__()
        self.l1 = nn.Linear(784,128)
        self.l2 = nn.Linear(128,10)
    def forward(self,x):
        a1 = self.l1(x)
        x1 = F.relu(a1)
        a2 = self.l2(x1)
        return a2
```

输入层:  
nn.Linear(784, 128): 全连接层  
784 = 28×28 (MNIST图像展平后的维度)  
128个神经元，可学习参数: (784+1)×128 = 100,480个  
  
激活函数:  
F.relu(): 修正线性单元  
公式: f(x) = max(0, x)  
引入非线性，使网络能够学习复杂模式  

输出层:  
nn.Linear(128, 10): 10个输出对应10个数字类别  
可学习参数: (128+1)×10 = 1,290个  
没有使用softmax: 因为F.cross_entropy内部会自动处理  
总参数量: 100,480 + 1,290 = 101,770个参数  

5. 使用SGD优化器，并指定学习率为0.1


```python
model = mlp().to(device)
optimizer = optim.SGD (model.parameters(),lr=0.01)
model
```




    mlp(
      (l1): Linear(in_features=784, out_features=128, bias=True)
      (l2): Linear(in_features=128, out_features=10, bias=True)
    )



model = MLP().to(device): 创建模型实例并移动到指定设备  
optim.SGD(): 随机梯度下降优化器  
lr=0.1: 相对较高的学习率，适用于批量较大(2048)的情况，高学习率可能收敛快，但可能不稳定

6. 循环训练与测试评估


```python
epochs = 30
for epoch in range(epochs):
    #训练
    model.train()
    for batch_idx,(x,y) in enumerate(train_loader):
        x,y = x.view(x.shape[0],-1).to(device),y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output,y)
        loss.backward()
        optimizer.step()
    #评估
    model.eval()
    correct = 0
    test_loss =0
    with torch.no_grad():
        for batch_idx,(x,y) in enumerate(test_loader):
            x,y = x.view(x.shape[0],-1).to(device),y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output,y)
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
        test_loss = test_loss/(batch_idx+1)
        acc = correct/len(test_loader.dataset)
        print ('epoch:{}, loss:{:.4f}, acc:{:.4f}'.format(epoch,test_loss, acc))
```

    epoch:0, loss:2.2512, acc:0.1936
    epoch:1, loss:2.1957, acc:0.4300
    epoch:2, loss:2.1329, acc:0.6026
    epoch:3, loss:2.0622, acc:0.6827
    epoch:4, loss:1.9828, acc:0.7208
    epoch:5, loss:1.8958, acc:0.7413
    epoch:6, loss:1.8008, acc:0.7591
    epoch:7, loss:1.7005, acc:0.7714
    epoch:8, loss:1.5973, acc:0.7808
    epoch:9, loss:1.4946, acc:0.7890
    epoch:10, loss:1.3948, acc:0.7959
    epoch:11, loss:1.3000, acc:0.8034
    epoch:12, loss:1.2134, acc:0.8095
    epoch:13, loss:1.1347, acc:0.8155
    epoch:14, loss:1.0648, acc:0.8226
    epoch:15, loss:1.0021, acc:0.8291
    epoch:16, loss:0.9468, acc:0.8326
    epoch:17, loss:0.8975, acc:0.8379
    epoch:18, loss:0.8542, acc:0.8434
    epoch:19, loss:0.8162, acc:0.8479
    epoch:20, loss:0.7823, acc:0.8504
    epoch:21, loss:0.7518, acc:0.8517
    epoch:22, loss:0.7241, acc:0.8543
    epoch:23, loss:0.6999, acc:0.8567
    epoch:24, loss:0.6772, acc:0.8597
    epoch:25, loss:0.6572, acc:0.8614
    epoch:26, loss:0.6390, acc:0.8634
    epoch:27, loss:0.6218, acc:0.8648
    epoch:28, loss:0.6070, acc:0.8684
    epoch:29, loss:0.5923, acc:0.8699
    

model.train(): 设置为训练模式，启用dropout和batch normalization  
数据预处理:  
x.view(x.shape[0], -1): 将图像从(batch,1,28,28)展平为(batch,784)  
.to(device): 将数据移动到指定设备  
optimizer.zero_grad(): 清空梯度，防止梯度累积  
前向传播: output = model(x)计算预测值  
损失计算: F.cross_entropy(output, y)，结合了softmax和负对数似然损失，适用于多分类问题  
反向传播: loss.backward()计算梯度  
参数更新: optimizer.step()更新模型参数  

model.eval(): 设置为评估模式，禁用dropout等训练专用层  
torch.no_grad(): 禁用梯度计算，节省内存和计算资源  
预测计算:  
output.max(1, keepdim=True)[1]: 获取最大概率的类别索引  
dim=1表示在类别维度上取最大值  
准确率计算:  
pred.eq(y.view_as(pred)): 比较预测和真实标签  
.sum().item(): 统计正确预测的数量  

损失归一化: test_loss / (batch_idx + 1)计算平均损失  
准确率计算: correct / len(test_loader.dataset)计算整体准确率  
格式化输出: 显示epoch、损失和准确率  

从结果可以发现第一轮训练后，准确率为19.36%，说明模型处于初始学习阶段。随后，准确率稳步提升，损失持续下降，最后在第29轮达到86.99%。损失也从2.2512下降到0.5923。准确率对于MNIST数据集来说并不算高，它通常可以达到97%以上，这是因为模型比较简单（仅含一个128节点的隐藏层），且使用了较大的Batch Size和未进行充分调优的SGD优化器。

## 实验小结

本次实验，用PyTorch实现了一个多层感知机，并在MNIST手写数字数据集上完成了图像分类任务。掌握了数据加载、张量转换、模型构建、训练和测试的基本流程，进一步理解了全连接层、ReLU激活函数、交叉熵损失函数以及随机梯度下降优化器在神经网络中的作用。  
实验中，将$28 \times 28$的图像展平为784维向量，输入包含一个128个神经元隐藏层的模型，最终输出10个类别的得分。训练过程中，通过反向传播计算梯度并更新模型参数；测试过程中，关闭梯度计算，根据输出得分最高的类别得到预测结果。  
从实验结果来看，训练30轮后，测试准确率由第一轮结束时的19.36%提升至86.99%，测试损失由2.2512下降至0.5923，说明模型能够从数据中学习手写数字的特征。不过，最终准确率仍有提升空间。模型结构、批量大小、学习率和训练轮数等都可能影响结果，具体影响需要通过对比实验验证。后续可以尝试调整这些参数，比较不同优化器的效果，并观察错误分类的样本，以进一步分析模型的不足。  
