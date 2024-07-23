import torch
import pandas as pd
import numpy as np
from torch import nn
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
# 激活器改为了Adam激活器，激活函数选择Relu激活函数最高效，在单隐藏层后加入了dropout，相应的丢弃概率为0.15



############泰坦尼克号数据集读取与预处理
data = pd.read_csv(r"D:\Document\pycharm\UniqueAI\week2_mlp\泰坦尼克号数据\泰坦尼克号数据.csv")


train_data1 = data.iloc[:, 2:3]
train_data2 = data.iloc[:, 4:5]
train_data3 = data.iloc[:, 5:8]
train_data4 = data.iloc[:, 9:10]
train_data5 = data.iloc[:, 11:]

#缺失值处理：均值填充Age列数据缺失值
train_data3.fillna(train_data3.mean(),inplace=True)

#特征值：特征值合并
train_data_feature = pd.concat((train_data1, train_data2, train_data3, train_data4, train_data5), axis=1)
#标签值：取出标签值
all_labels = data.iloc[:, 1:2]

#独热编码：对特征值进行独热编码并转换为numpy数组类型
all_features = pd.get_dummies(train_data_feature)
all_features = all_features.to_numpy()
all_labels = all_labels.to_numpy()



#数据类型转换：将独热编码生成的布尔类型转换为浮点数类型
all_features = all_features.astype(float)
all_labels = all_labels.astype(float)

#将标签值转换为一维向量，是为了与前面的梯度下降算法实现对应
all_labels = all_labels.flatten()
############



# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.15)

# 把数据都转化为numpy的narray数组类型
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
#统一数据的小数点类型
X_train = X_train.to(torch.float)
X_test = X_test.to(torch.float)
y_train = y_train.long()
y_test = y_test.long()


######################################## 定义模型并初始化
net = nn.Sequential(nn.Linear(10, 16),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(16, 2))


# 参数初始化函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)



#########################训练函数定义
# 数据迭代器函数
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)


# 对准确率的评估函数
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    # X, y = data_iter:
    X = data_iter[0]
    y = data_iter[1]
    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
    n += y.shape[0]

    return acc_sum / n



# 具体训练函数同时预测测试集准确率
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            # 执行优化方法
            if optimizer is not None:
                optimizer.step()

            else:
                d2l.sgd(params, lr, batch_size)

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            # print(n)
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


# 训练开始
batch_size, lr, num_epochs = 128, 0.0002, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay= 3)


train_iter = load_array((X_train, y_train), batch_size)


train_ch3(net, train_iter, (X_test, y_test), loss, num_epochs, batch_size, lr= lr, optimizer=trainer)
