import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib



# 设置字符串,防止中文乱码
mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False

# ######################################
# 加载数据以及数据清洗
# ######################################
df = pd.read_csv("D:\人工智能\数据集\energydata_complete_1000.csv")
# print(df.head(3))
X1 = df['date']
X1 = X1.apply(lambda x: pd.Series(time.strptime(''.join(x), '%Y/%m/%d  %H:%M')))
X1 = X1.iloc[:, 2:5]
X = df.iloc[:, 2:20]
Y = df['Appliances']
# print(X.head())
# ######################################
# 数据分割
# ######################################
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75)


# ######################################
# 构造特征工程
# ######################################

'''
方案一:使用标准拟合办法
ss = StandardScaler()
# print(x_train)
X_train = ss.fit_transform(x_train)
lr = LinearRegression(fit_intercept=False)
lr.fit(X_train, y_train)
y_pred = lr.predict(x_test)

print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(lr.coef_))
print('训练集上误差为:', lr.score(X_train, y_train))
print('测试集上误差为:', lr.score(x_test, y_test))
'''

'''
# 方案二:使用多项式回归的办法
class Poly:
      def __init__(self, n):
            self.degree = n

      def poly(self, x_train, x_test, y_train):
            ss = PolynomialFeatures(self.degree)
            # print(x_train)
            X_train = ss.fit_transform(x_train)
            x_test = ss.fit_transform(x_test)
            lr = LinearRegression(fit_intercept=False)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(x_test)
            error_train, error_test = lr.score(X_train, y_train), lr.score(x_test, y_test)

            return error_train,error_test, y_pred
'''
# 方案一:使用标准拟合办法
ss = StandardScaler()
# print(x_train)
X_train = ss.fit_transform(x_train)
lr = LinearRegression(fit_intercept=False)
lr.fit(X_train, y_train)
# y_pred = lr.predict(x_test)

print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(lr.coef_))
print('训练集上误差为:', lr.score(X_train, y_train))
print('测试集上误差为:', lr.score(x_test, y_test))



plt.plot(x_train, y_train)