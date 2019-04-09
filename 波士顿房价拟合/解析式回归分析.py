import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split

# 1. 加载数据(数据一般存在于磁盘或者数据库)
data = pd.read_csv(r"D:\人工智能\波士顿房价数据.csv")
# print(data.info())
# data.split('  ')
# print(data.info())
# 2. 数据清洗

# 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = data.iloc[:, 0:13]
Y = data.iloc[:, 13]

# print(X.head())
# print(Y.head())

# 4. 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y)

# 5. 特征工程的操作
# print('转换前:',x_train)
x_train = np.mat(x_train)
y_train = np.mat(y_train).reshape(-1, 1)
# print('转换后:',x_train.shape)

# 6. 模型对象的构建
thea = (x_train.T * x_train).I * x_train.T * y_train
# print(thea)
# 7. 模型的训练
x_test = np.mat(x_test)
# y_test = np.mat(y_test)

y_pred = x_test * thea
# print(len(y_pred))
# print('**'*20)
print(y_test)
# 8. 模型效果评估
# print(np.mean(y_pred - y_test.T))
# print('均方差为:', R)
t = np.arange(len(x_test))
plt.plot(t, y_test, 'r', linewidth=2, label=u'真实值')
plt.plot(t, y_pred, 'b', linewidth=2, label=u'真实值')
plt.xlabel('波士顿房价的各个样本')
plt.ylabel('房价')
plt.title('波士顿房价预测')
plt.legend(loc='upper right')
plt.show()
# 9. 模型保存\模型持久化