import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

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
# print('转换前:',x_train.shape)
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
# y_train = np.mat(y_train).reshape(-1, 1)
# print('转换后:',x_train.shape)

# 6. 模型对象的构建
# thea = (x_train.T * x_train).I * x_train.T * y_train
# print(thea)
# 7. 模型的训练

model = Pipeline([
    ('Poly', PolynomialFeatures()),  # 给定进行多项式扩展操作， 第一个操作：多项式扩展
    ('Linear', LinearRegression(fit_intercept=False))  # 第二个操作，线性回归
])
# print(len(y_pred))
# print('**'*20)
# print(y_test)
# 8. 模型效果评估
x_test = ss.transform(x_test)
y_pred = lr.predict(x_test)
print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(lr.coef_))
print("截距项值:{}".format(lr.intercept_))
# b. 直接通过评估相关的API查看效果
print("模型在训练数据上的效果(R2)：{}".format(lr.score(x_train, y_train)))
# 在测试的时候对特征属性数据必须做和训练数据完全一样的操作
# x_test = ss.transform(x_test)
print("模型在测试数据上的效果(R2)：{}".format(lr.score(x_test, y_test)))
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