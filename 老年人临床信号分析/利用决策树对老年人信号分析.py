import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# from ML.linear_regression import R2_square
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
pd.read_csv(i, header=None, names=names, sep=',') for i in data_path
]
df = pd.concat(df)
df.replace('?', np.nan, inplace=True)
df = df.dropna(axis=0, how='any')

# 数据分割
X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]

tree_1 = X.vertical_axis
# print(tree_1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=20)

# 建立模型
algo = DecisionTreeClassifier(criterion="gini", max_depth=None,
                              min_samples_leaf=1, min_samples_split=2, min_impurity_decrease=0.0008)
algo.fit(x_train, y_train)
y_pred = algo.predict(x_test)

# 模型预测
print('训练集误差为：', algo.score(x_train, y_train))
print('测试集误差为：', algo.score(x_test, y_test))

# 模型展示
t = np.arange(len(x_test))

plt.plot(t, y_pred, 'r')
plt.plot(t, y_test, 'b')

# b. 可视化方式二：直接使用pydotpuls库将数据输出为pdf或者png格式
from sklearn import tree
import pydotplus

dot_data = tree.export_graphviz(decision_tree=algo, out_file=None,
                                feature_names=feature,
                                class_names=['sit on bed', 'sit on chair', 'lying', 'ambulating'],
                                filled=True, rounded=True,
                                special_characters=True
                                )

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('older_entropy.pdf')