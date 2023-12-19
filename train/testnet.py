import torch
from torch import nn

# 定义输入/输出元素个数
inputnums = 6
outputnums = 9

# 判断GPU是否可用
use_gpu = torch.cuda.is_available()

# 定义超参数
batch_size = 512

# 定义简单的前馈神经网络
class neuralNetwork5Layer(nn.Module):
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

pthfile = r'./neural_network_singlecom_10000.pth'
model.load_state_dict(torch.load(pthfile))
print(model)


# 使用DataLoader方法进行批量读取数据
from torch.utils.data import DataLoader
import numpy as np
test_dataset = [[-1.04719, 0.34907, 1.57080, 0, -1.91986, 0]]
test_dataset = np.array(test_dataset)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=True)

model.eval()
for data in test_loader:
    features = data[:,0:inputnums]
    features = features.float()
    if use_gpu:
        features = features.cuda()
    with torch.no_grad():
        out = model(features)
        print("out =", out)

real = [[0.5, 0, -0.866025, 0, 1, 0, 0.8660254, 0, 0.5]]
print("real =", real)