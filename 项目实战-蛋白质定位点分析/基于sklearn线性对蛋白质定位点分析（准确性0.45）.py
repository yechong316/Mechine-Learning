'''

http://archive.ics.uci.edu/ml/datasets/Ecoli
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import LabelEncoder
def R2_score(y_true, y_pred):
    assert y_pred.shape == y_true.shape, '输入的{}与{}不相等'.format(y_pred.shape, y_true.shape)
    return np.sum(np.square(y_pred - y_true)) / np.sum(np.square(y_pred - np.mean(y_true)))

# 数据加载
names = ['mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'classtic']
df = pd.read_csv('./datas/ecoli.data', sep='  ', names=names)
le = LabelEncoder()
X = df.iloc[:, 0:-1]
Y = pd.DataFrame(le.fit_transform(df.iloc[:, -1]))

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y)

# print('训练集的数据格式为{}'.format(x_train.shape))
# print('测试集的数据格式为{}'.format(y_train.shape))
x_train = np.mat(x_train)
y_train = np.mat(y_train).reshape(-1, 1)

# 特征工程
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
lr = LinearRegression()
lr.fit(x_train, y_train)




# 模型预测
x_test = ss.transform(x_test)
y_pred_test = lr.predict(x_test)
# print('训练集的数据格式为{}'.format(x_test.shape))
# print('测试集的数据格式为{}'.format(y_test.shape))

# print('训练集的方差为{}'.format(R2_score(y_pred_train, y_train)))
print('测试集的方差为{}'.format(lr.score(x_test, y_test)))
print('训练集的方差为{}'.format(lr.score(x_train, y_train)))

# 模型展示
t = np.arange(len(x_test))
plt.plot(t, y_test, 'r', label=u'实际值')
plt.plot(t, y_pred_test, 'b', label=u'预估值')
plt.show()
