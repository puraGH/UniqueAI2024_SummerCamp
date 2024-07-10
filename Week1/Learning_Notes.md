# Week1

---



# 缺失值的处理

## 1. 数据缺失的原因、类型以及处理方法



![缺失值处理](https://github.com/puraGH/UniqueAI2024SummerCamp/blob/xiayu/Week1/%E7%BC%BA%E5%A4%B1%E5%80%BC%E5%A4%84%E7%90%86.png?raw=true)





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







