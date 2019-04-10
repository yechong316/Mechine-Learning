import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 数据加载
data = pd.read_csv(r"D:\旧电脑\人工智能\05_随堂代码\[20181104_1110_1111]_回归算法\datas\winequality-white.csv")
# print(data.info())
## 自变量名称
names = ["fixed acidity","volatile acidity","citric acid",
         "residual sugar","chlorides","free sulfur dioxide",
         "total sulfur dioxide","density","pH","sulphates",
         "alcohol", "type"]

## 因变量名称
quality = "quality"
# print(data.info())
X = data.iloc[:, 0:-1]
# print(X)/
Y = data[quality]
# # print(x.head())
# print(y)
# 数据清洗

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y)

# 数据训练
algo  = Pipeline(steps=[
    ('ss', StandardScaler()),
    ('algo', LinearRegression())
])
# x_train = pipeline.fit_transform(x_train)
algo.fit(x_train, y_train)

# 数据结果
print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(algo.get_params()['algo'].coef_))
print("截距项值:{}".format(algo.steps[-1][1].intercept_))
# b. 直接通过评估相关的API查看效果
print("模型在训练数据上的效果(R2)：{}".format(algo.score(x_train, y_train)))
# 在测试的时候对特征属性数据必须做和训练数据完全一样的操作
print("模型在测试数据上的效果(R2)：{}".format(algo.score(x_test, y_test)))


# 数据展示

# 数据保存