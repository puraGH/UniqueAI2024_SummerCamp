# 多层感知机

## 基本概念

​	多层感知机（Multilayer Perceptron，简称MLP），是一种基于前馈神经网络（Feedforward Neural Network）的深度学习模型，由多个神经元层组成，其中每个神经元层与前一层全连接。多层感知机可以用于解决分类、回归和聚类等各种机器学习问题。
​	多层感知机的每个神经元层由许多神经元组成，其中输入层接收输入特征，输出层给出最终的预测结果，中间的隐藏层用于提取特征和进行非线性变换。每个神经元接收前一层的输出，进行加权和和激活函数运算，得到当前层的输出。通过不断迭代训练，多层感知机可以自动学习到输入特征之间的复杂关系，并对新的数据进行预测。

相较于单层感知机，多层感知机的改进如下：

1. 引入了隐藏层(hidden layer)的结构，隐藏层通常指代的是，输入层(input layer)和输出层(output layer)中间的具有 N 个神经元的结构。其中层与层之间采用全连接的结构，跨层之前没有相连。
2. 引入了新的非线性激活函数。
3. 采用了反向传播算法(back propagation)。

![前向传播](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week2/mlp%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD.png?raw=true)

![反向传播](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week2/mlp%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD.png?raw=true)



## Adam优化器

​	Adam（Adaptive Moment Estimation）优化器是一种自适应优化算法，可以根据历史梯度信息来调整学习率。它结合了RMSProp和Momentum两种优化算法的思想，并且对参数的更新进行了归一化处理，使得每个参数的更新都有一个相似的量级，从而提高训练效果。Adam优化器在很多实际问题中表现良好，尤其是在大规模数据集上训练深度神经网络时效果更佳。

Adam的推导公式：

![Adam推导公式](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week2/Adam%E6%8E%A8%E5%AF%BC%E5%85%AC%E5%BC%8F.png?raw=true)



## dropout

​	dropout 是指在深度网络的训练中, 以一定的概率随机地 “临时丢弃” 一部分神经元节点. 具体来讲, dropout 作用于每份小批量训练数据，以一定的概率随机丢弃掉部分节点, 由于其随机丢弃部分神经元的机制, 相当于每次迭代都在训练不同结构的神经网络.主要用于缓解模型训练过拟合的问题

​	dropout 前网络结构示意 :

![dropout1](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week2/dropout%E5%89%8D%E7%BD%91%E7%BB%9C.png?raw=true)

​	dropout 后网络结构示意 :

![dropout2](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week2/dropout%E5%90%8E%E7%BD%91%E7%BB%9C.png?raw=true)



---

# CNN卷积神经网络

​	卷积神经网络（Convolutional Neural Networks，CNN）是一类包含卷积计算且具有深度结构的前馈神经网络（Feedforward Neural Networks），广泛应用于图像处理、自然语言处理等领域。是深度学习（deep learning）的代表算法之一。

​	LeNet网络简介： LeNet-5共有7层，不包含输入，每层都包含可训练参数；每个层有多个Feature Map，每个Feature Map是通过一种卷积滤波器提取输入的一种特征，然后每个Feature Map有多个神经元。CNN最基本的架构：卷积层、池化层、全连接层。

![lenet网络](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week2/lenet%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84.png?raw=true)





