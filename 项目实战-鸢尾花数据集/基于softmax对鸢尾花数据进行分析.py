import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 数据加载
names = ['sepal length', 'sepal_width', 'petal_length', 'petal_width', 'cla']
df = pd.read_csv(r'D:\人工智能\mechine_learning\数据集\iris.data', sep=',',header=None, names=names)
# print(Counter(data.cla).most_common())

df.replace('Iris-setosa', 1,inplace=True)
df.replace('Iris-virginica', 2,inplace=True)
df.replace('Iris-versicolor', 3,inplace=True)
# print(df['cla'].value_counts())
# print(df.head())
X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]
# print('X:', X)
# print('Y:', Y)

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y)

# 数据分析
algo = LogisticRegression()
algo.fit(x_train, y_train)

# 结果分析
print('训练集上的R2为:{}'.format(algo.score(x_train, y_train)))
print('测试集上的R2为:{}'.format(algo.score(x_test, y_test)))

# 结果展示
y_pred = algo.predict(x_test)
fig = plt.figure(figsize=(10, 10), facecolor='w')
t = np.arange(len(x_test))
plt.scatter(t, y_test,c='b')
plt.scatter(t, y_pred,c='r')
plt.show()