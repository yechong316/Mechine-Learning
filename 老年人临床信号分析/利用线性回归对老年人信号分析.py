import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from ML.linear_regression import R2_square
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 数据加载
names = [
    'Time_second', 'frontal_axis', 'vertical_axis',
    'lateral_axis', 'Id', 'RSSI', 'Phase', 'Frequency', 'activity',
]

feature = [
    'Time_second', 'frontal_axis', 'vertical_axis',
    'lateral_axis', 'Id', 'RSSI', 'Phase', 'Frequency'
]
filefold_path = '../datas/S1_Dataset'
# print(os.listdir(filefold_path))

data_path = [
    filefold_path + '/' + i for i in os.listdir(filefold_path)
]
df = [
pd.read_csv(i, header=None, names=names,sep=',') for i in data_path
]
# df2 = pd.read_csv(data_path[1], header=None, names=names,sep=',')
# print(df1.head())
# print(df2.head())
df = pd.concat(df)
# print(df.head())
# print(df.info())
df.replace('?', np.nan, inplace=True)
# 删除为nan的数据
# axis：指定按照什么维度来删除数据，0表示第一维，也就是DataFrame中的行。1表示列
# how：指定进行什么样的删除操作，any表示只要出现任意一个特征属性为nan，那么就删除当前行或者当前列。all表示只有当所有的特征属性值均为nan的时候，才删除当前行或者当前列
df = df.dropna(axis=0, how='any')
# print(df.info())

# 数据分割
X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=20)


# 建立模型
poly = PolynomialFeatures()
x_train = poly.fit_transform(x_train)
x_test = poly.transform(x_test)

algo = LinearRegression()
algo.fit(x_train, y_train)

# 模型预测
print('训练集误差为：', algo.score(x_train, y_train))
print('测试集误差为：', algo.score(x_test, y_test))

y_pred = algo.predict(x_test)