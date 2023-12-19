import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from datetime import datetime

# 定义超参数
batch_size = 512
learning_rate = 0.01
num_epochs = 100 # 训练次数

# 定义输入/输出元素个数
inputnums = 6
outputnums = 9

# 判断GPU是否可用
use_gpu = torch.cuda.is_available()
print(torch.__version__)
print(torchvision.__version__)
print(torch.version.cuda)
print("isGPU =", use_gpu)

# 读取csv文件
import pandas as pd
# dataset = pd.read_csv('dataset.csv')
dataset = pd.read_csv('singlecomponent.csv')

# 将数据转化成一个去掉表头的标准numpy二维数组
dataset = dataset.values

# 设置训练集和测试集的长度
train_size = int(len(dataset) * 0.8)  # 这里按照8：2进行训练和测试
test_size = len(dataset) - train_size

# 打乱数据后按比例分割
import torch.utils.data as Data
train_dataset, test_dataset = Data.random_split(dataset, [train_size, test_size])

# 使用DataLoader方法进行批量读取数据
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=True)

# 分割样本特征features以及样本的标签tag
# for data in train_loader:
#     featurs = data[:, :-1]
#     tags = data[:, -1:]
	
# 定义简单的前馈神经网络
class neuralNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(neuralNetwork, self).__init__() # super() 函数是用于调用父类(超类)的一个方法
# Sequential()表示将一个有序的模块写在一起，也就相当于将神经网络的层按顺序放在一起，这样可以方便结构显示
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.Sigmoid()) # 表示使用Sigmoid激活函数
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.Sigmoid())
        self.outputlayer = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim))

# 定义向前传播
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.outputlayer(x)
        return x

# 定义简单的前馈神经网络
# 为适应cpp做了更改
# class neuralNetwork5Layer(nn.Module):
class neuralNetwork5Layer(torch.jit.ScriptModule):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(neuralNetwork5Layer, self).__init__() # super() 函数是用于调用父类(超类)的一个方法
# Sequential()表示将一个有序的模块写在一起，也就相当于将神经网络的层按顺序放在一起，这样可以方便结构显示
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.Sigmoid()) # 表示使用Sigmoid激活函数
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.Sigmoid())
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.Sigmoid()) # 表示使用Sigmoid激活函数
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_4),
            nn.Sigmoid())
        self.layer5 = nn.Sequential(
            nn.Linear(n_hidden_4, n_hidden_5),
            nn.Sigmoid())
        self.outputlayer = nn.Sequential(
            nn.Linear(n_hidden_5, out_dim))

# 定义向前传播
# 为适应cpp加了注解
    @torch.jit.script_method
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.outputlayer(x)
        return x

# 层节点数目（包括输入层、输出层）
model = neuralNetwork5Layer(inputnums, 80, 80, 80, 80, 80, outputnums)
if use_gpu:
    model = model.cuda() # 现在可以在GPU上跑代码了

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()   #没有需要保存的参数和状态信息
        
    def forward(self, x, y):  # 定义前向的函数运算即可
        return torch.mean(torch.pow((x - y), 2))
    
criterion = nn.MSELoss() # 定义损失函数类型
# criterion = My_loss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # 定义优化器，使用随机梯度下降
# optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0) #不收敛
# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0) #收敛 但是下降速度很慢
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) #收敛 速度快 效果不错的
optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) #收敛 速度快 效果不错的
# optimizer = torch.optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0) #不收敛
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False) #收敛 效果也还可以
# optimizer = torch.optim.Rprop(model.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50)) #收敛的很慢 效果不太好

starttime = datetime.now() # 获得当前时间
# 开始模型训练
for epoch in range(num_epochs):
    # print('*' * 10)
    # print(f'epoch {epoch+1}')
    running_loss = 0.0 # 初始值
    running_acc = 0.0 # 准确率，暂时不算
    for i, data in enumerate(train_loader, 1): # 枚举函数enumerate返回下标和值
        features = data[:,0:inputnums]
        tags = data[:,inputnums:]
        features = features.float()
        tags = tags.float()
        # 使用GPU？
        if use_gpu:
            features = features.cuda()
            tags = tags.cuda()
        # 向前传播
        out = model(features) # 前向传播
        # print(out.size())
        # print(tags.size())
        # print("out = ", out)
        # print("tags = ", tags)
        loss = criterion(out, tags) # 计算loss
        running_loss += loss.item() # loss求和
        # 向后传播
        optimizer.zero_grad() # 梯度归零
        loss.backward() # 后向传播
        optimizer.step() # 更新参数

        # if i % 300 == 0:
            # print(f'[{epoch+1}/{num_epochs}] Loss: {running_loss/i:.6f}, Acc: {running_acc/i:.6f}')
    if epoch % 20 == 0: # 每20轮报告一轮
        nowtime = datetime.now();
        spendtime = (nowtime - starttime).seconds
        print(f'Finish {epoch+1} epoch, Loss: {running_loss/i:.6f}, Acc: {running_acc/i:.6f}, Time: {spendtime}')
    
# 模型测试
model.eval() # 让模型变成测试模式
eval_loss = 0.
eval_acc = 0.
for data in test_loader:
    features = data[:,0:inputnums]
    tags = data[:,inputnums:]
    features = features.float()
    tags = tags.float()
    if use_gpu:
        features = features.cuda()
        tags = tags.cuda()
    with torch.no_grad():
        out = model(features)
        # print("out = ", out)
        # print("tags = ", tags)
        loss = criterion(out, tags)
    eval_loss += loss.item()
print(f'Test Loss: {eval_loss/len(test_loader):.6f}, Acc: {eval_acc/len(test_loader):.6f}\n')

# 保存模型
# torch.save(model.state_dict(), './neural_network_singlecom_10000.pth')
# torch.jit.save(model, './neural_network_singlecom_test.pt')
model.save('./neural_network_singlecom_test_torch170_cu101.pt')
