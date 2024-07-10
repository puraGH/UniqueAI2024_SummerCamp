# UniqueAI夏令营第一周任务

## 前置任务
1.配置linux环境，windows使用双系统或者使用wsl（建议使用wsl），mac用户可以基本跳过此步。如果你恰巧财力雄厚，我们当然也非常欢迎你直接使用linux服务器。

2.配置好git环境并在本仓库下建立自己的分支比提交第一个pr。

3.配置python环境，建议使用anaconda或者miniconda。

请使以在markdown中提交截图的方式提交该任务。

## 数据处理
数据处理是AI中基础且重要的一环，每个学习AI的同学应当首先学习数据处理的基本技巧。

数据集：[泰坦尼克号](https://uniquestudio.feishu.cn/drive/folder/fldcnV0PzAB5J8ZaoMp8WXho8if?from=from_copylink)

### 基础任务
1.缺失值处理，包括但不限于，knn填补，众数，均值填补，补零；

2.数据标准化，归一化;

3.对于离散型变量的OneHotEncoder。

### 进阶任务
使用torch中的Dataset和DataLoader类对上面处理过的数据集进行加载。

## 机器学习
可以用的py库：numpy, pandas, matplotlib，gym，time
禁止使用pytorch，tensorflow等深度学习库，禁用sklearn等直接调用模型的库。

### 基础任务
实现一个逻辑回归代码，数据集是上面的泰坦尼克数据集，要求通过给出的数据拟合是否生还。

### 进阶任务
1.尝试实现避免过拟合的方法；

2.尝试使用多种优化方法，包括但不限于SGD、Adam等；

3.尝试实现dropout。