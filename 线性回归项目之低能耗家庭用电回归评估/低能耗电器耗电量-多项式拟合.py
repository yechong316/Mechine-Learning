import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import seaborn as sns

from sklearn.model_selection import  train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 设置字符串,防止中文乱码
mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False

# ######################################
# 加载数据以及数据清洗
# ######################################
df = pd.read_csv("D:\人工智能\数据集\energydata_complete_1000.csv")
data = df.iloc[:, 1:29]

# ######################################
# 数据相关性分析
# ######################################
# plt.figure(figsize=(10, 10))
# sns.heatmap(data.corr(), annot=True)
# plt.show()
'''
热力图显示,耗电量与灯光强相关
'''

# ######################################
# 数据分割
# ######################################
X = df.iloc[:, 2:4]
Y = df['Appliances']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.6)

# ######################################
# 多项式拟合
# ######################################
scaler = PolynomialFeatures(degree=7)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# x_test = x_test.reshape((-1, 1))
# print('转换后训练集数据为:', x_train.shape)
lr = ElasticNet()
lr.fit(x_train, y_train)

# ###############################
# 误差训练
# ###############################
print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(lr.coef_))
print('训练集上误差为:', lr.score(x_train, y_train))
print('测试集上误差为:', lr.score(x_test, y_test))

# ###############################
# 结果展示
# ###############################
t = np.arange(len(x_test))
y_pred = lr.predict(x_test)
plt.plot(t, y_test, 'b', linewidth=2, label= '真实值')
plt.plot(t, y_pred, 'r', linewidth=2, label='预测值')
plt.legend(loc='upper right')
plt.title('低能耗家用电器耗电量与亮度的关系')
plt.xlabel('lights')
plt.ylabel('Appliances')
plt.grid()
plt.show()

