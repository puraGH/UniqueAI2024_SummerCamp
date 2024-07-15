# Week1

---



# 缺失值的处理

## 1. 数据缺失的原因、类型以及处理方法



![缺失值处理](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week1/%E7%BC%BA%E5%A4%B1%E5%80%BC%E5%A4%84%E7%90%86.png?raw=true)





## 2. pandas中None或NaN代表缺失值，检测缺失值的常用方法包括isnull，nonull，isna，notna

| 方法     | 说明                              |
| -------- | --------------------------------- |
| notna()  | 若返回的值为False，说明存在缺失值 |
| isnull() | 若返回的值为True，说明存在缺失值  |
| nonull() | 若返回的值为False，说明存在缺失值 |
| isna()   | 若返回的值为True，说明存在缺失值  |



## 3. 缺失值的处理方法

### 3.1  删除

一些缺失值比例过大的数据还是需要将其删除的，缺失值填补也仅仅只是基于当前数据进行预测，计算的，存在一定误差。但填补的数据过多，反而只会带来误差。

`del data['列名']`



### 3.2 填充固定值

将缺失值都填充为给定常数

`data.fillna({0:1000, 1:100, 2:0, 4:5}) `



### 3.3 填充中位数、平均数、众数

这三个数代码相近，只需要将fillna()函数里的value参数改为其他的即可。

`data.fillna(data.mean(),inplace=True) # 填充均值
data.fillna(data.median(),inplace=True) # 填充中位数
data.fillna(data.mode(),inplace=True) # 填充众数`



### 3.4 插值法填充，前值或者后值填充

利用插值方法填充缺失值是一种常见的缺失值处理技术，它可以根据已知数据点的数值来推断缺失值，并填充这些缺失值。

在 Pandas 中，可以使用 常用的interpolate() 方法或者fillna()方法来进行插值填充。

[interpolate函数方法使用]([Python pandas.DataFrame.interpolate函数方法的使用-CJavaPy](https://www.cjavapy.com/article/541/))

fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
#value：固定值，可以用固定数字、均值、中位数、众数等，此外还可以用字典，series等形式数据；
#method:填充方法，'bfill','backfill','pad','ffill'
#axis: 填充方向，默认0和index，还可以填1和columns
#inplace:在原有数据上直接修改
#limit:填充个数，如1，每列只填充1个缺失值

[fillna函数方法使用]([Python pandas.DataFrame.fillna函数方法的使用-CJavaPy](https://www.cjavapy.com/article/460/))



### 3.5 knn填充

基本思想： 先将数据标准化，然后对缺失值的数据点做k邻近填充，计算含缺失值的数据点与其他不含缺失值的数据点的距离矩阵，选出欧氏距离最近的k个数据点。用选中的k个近邻的数据点对应的字段均值来填充数据中的空缺值。

```python
from fancyimpute import KNN
data_train_knn = pd.DataFrame(KNN(k=6).fit_transform(data_train_shanchu)#这里的6是对周围6个数据进行欧式距离计算，得出缺失值的结果，可以自行调整
columns=data_train_shanchu.columns)
data_train_knn
```



# 数据的标准化与归一化

归一化和标准化都是对数据做变换的方式，将原始的一列数据转换到某个范围，或者某种形态

## 归一化

归一化(N o r m a l i z a t i o n NormalizationNormalization)：将一列数据变化到某个固定区间(范围)中，通常，这个区间是[0, 1]，广义的讲，可以是各种区间，比如映射到[0，1]一样可以继续映射到其他范围，图像中可能会映射到[0,255]，其他情况可能映射到[-1,1]；常见的包括均值归一化和最大最小值归一化。

均值归一化：

![均值归一化](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week1/%E5%9D%87%E5%80%BC%E5%BD%92%E4%B8%80%E5%8C%96.png?raw=true)

最大最小值归一化：

![最大最小值归一化](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week1/%E6%9C%80%E5%A4%A7%E6%9C%80%E5%B0%8F%E5%80%BC%E5%BD%92%E4%B8%80%E5%8C%96.png?raw=true)



## 标准化

标准化(Standardization)：将数据变换为均值为0，标准差为1的分布切记，并非一定是正态的；

中心化：另外，还有一种处理叫做中心化，也叫零均值处理，就是将每个原始数据减去这些数据的均值。

![标准化](https://github.com/puraGH/UniqueAI2024_SummerCamp/blob/main/Week1/%E6%95%B0%E6%8D%AE%E6%A0%87%E5%87%86%E5%8C%96.png?raw=true)



```python
#最大最小值归一化
def MaxMinNormalization(data,min,max):
    data = (data - min) / (max - min)
    return data
#标准化
def ZscoreNormalization(data, mean, std):
    data= (data - mean) / std
    return data
```



# OneHotEncoder独热编码

​	独热码，在英文文献中称做 one-hot code, 直观来说就是有多少个状态就有多少比特，而且只有一个比特为1，其他全为0的一种码制。

​	我们在进行建模时，变量中经常会有一些变量为离散型变量，例如性别。这些变量我们一般无法直接放到模型中去训练模型。因此在使用之前，我们往往会对此类变量进行处理。一般是对离散变量进行one-hot编码。one hot encoder能将离散特征转化为二进制向量特征的函数，二进制向量每行最多有一个1来表示对应的离散特征某个值。

​	有两种哑编码的实现方法，[pandas](https://so.csdn.net/so/search?q=pandas&spm=1001.2101.3001.7020)和sklearn。它们最大的区别是，pandas默认只处理字符串类别变量，sklearn默认只处理数值型类别变量(需要先 LabelEncoder )

pandas：
```python
import pandas as pd  
  
#data的数据类型必须是DataFrame, Series或array-like
dummies = pd.get_dummies(data)  
print(dummies)
```



# 逻辑回归算法

​	线性回归主要用于预测连续的数值输出，基于线性关系模型，其目标是最小化实际值和预测值之间的差异。
​	逻辑回归主要用于分类问题，尤其是二元分类，它预测属于某一类别的概率，并基于概率输出进行决策，使用的是逻辑（Sigmoid）函数将线性模型的输出转换为概率值。简单说就是：找到一组参数，使得模型对分类结果的预测概率最大化。

​	举个例子：在预测银行贷款这件事上，线性回归可以帮你预测银行能发放的贷款额度是多少，逻辑回归则是尽可能准确地预测银行能否发放贷款（要么0，要么1）。

​	在线性回归中，我们着力于在U型曲线里找到局部最低点，以最小损失形成模型去预测结果。而在逻辑回归中，我们着力于最大化似然函数，这通常意味着找到一组参数，使得给定数据集中观测到的分类结果出现的概率最大。这个过程可以视为通过调整模型参数来提高正确分类的概率，从而在概率空间中寻找最优解。简单说：在逻辑回归中，我们致力于找到参数设置，使得模型对数据进行正确分类的可能性最大化。

[逻辑回归算法原理]([逻辑回归原理及代码_逻辑回归代码-CSDN博客](https://blog.csdn.net/eevee_1/article/details/134967433))



