import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 设置随机种子
seed_value = 2023
np.random.seed(seed_value)


# Sigmoid激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 定义逻辑回归算法
class LogisticRegression:
    def __init__(self, learning_rate=0.003, iterations=100):
        self.learning_rate = learning_rate  # 学习率
        self.iterations = iterations  # 迭代次数

    def fit(self, X, y):
        # 初始化参数
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0

        # 梯度下降
        for i in range(self.iterations):
            # 计算sigmoid函数的预测值, y_hat = w * x + b
            y_hat = sigmoid(np.dot(X, self.weights) + self.bias)


            # 计算损失函数
            loss = (-1 / len(X)) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


            # 计算梯度
            dw = (1 / len(X)) * np.dot(X.T, (y_hat - y))
            db = (1 / len(X)) * np.sum(y_hat - y)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 打印损失函数值
            if i % 10 == 0:
                print(f"Loss after iteration {i}: {loss}")

    # 预测
    def predict(self, X):
        y_hat = sigmoid(np.dot(X, self.weights) + self.bias)
        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        return y_hat

    # 精度
    def score(self, y_pred, y):
        accuracy = (y_pred == y).sum() / len(y)
        return accuracy



############泰坦尼克号数据集读取与预处理
data = pd.read_csv(r"D:\Document\pycharm\UniqueAI\week1\泰坦尼克号数据\泰坦尼克号数据.csv")

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


#归一化函数
def MaxMinNormalization(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data


#对特征值进行归一化
all_features = MaxMinNormalization(all_features)


#数据类型转换：将独热编码生成的布尔类型转换为浮点数类型
all_features = all_features.astype(float)
all_labels = all_labels.astype(float)

#将标签值转换为一维向量，是为了与前面的梯度下降算法实现对应
all_labels = all_labels.flatten()
############



# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.15, random_state=seed_value)


# 训练模型
model = LogisticRegression(learning_rate=0.03, iterations=1000)
model.fit(X_train, y_train)

# 结果
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

score_train = model.score(y_train_pred, y_train)
score_test = model.score(y_test_pred, y_test)

print('训练集Accuracy: ', score_train)
print('测试集Accuracy: ', score_test)
