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
X = df['date']
X = X.apply(lambda x: pd.Series(time.strptime(''.join(x), '%Y/%m/%d  %H:%M')))
X = X.iloc[:, 2:5]
Y = df['Appliances']
# print(X.head()
#       )
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



# ss = PolynomialFeatures(3)
# # print(x_train)
# X_train = ss.fit_transform(x_train)
# x_test = ss.fit_transform(x_test)
# lr = LinearRegression(fit_intercept=False)
# lr.fit(X_train, y_train)
# y_pred = lr.predict(x_test)


# print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(lr.coef_))
# print('训练集上误差为:', lr.score(X_train, y_train))
# print('测试集上误差为:', lr.score(x_test, y_test))


# ######################################
# 模型展示
# ######################################
list = [2, 3, 5, 7]
error_train, error_test,y_pred = [], [], []
for i in list:
      ploy_ins = Poly(i)
      e_train, e_test,y_hat = ploy_ins.poly(x_train, x_test, y_train)
      error_train.append(e_train)
      error_test.append(e_test)
      y_pred.append(y_hat)


print(error_test)
fig, axes = plt.subplots(4,1)
color = ['y', 'r', 'b', 'g']
t = np.arange(len(x_test))
for i in range(4):

    axes[i].plot(t, y_test, 'c', linewidth=2, label='真实值')
    axes[i].plot(t, y_pred[i], color[i], linewidth=2, label='预测值')
    # axes[i].title('线性回归之时间与耗电量之间的关系', fontsize=20)
    axes[i].set_xlabel('时间')
    axes[i].set_ylabel('耗电量')
    axes[i].grid(b=True)

t = np.arange(len(x_test))
# plt.figure(facecolor='w')
# plt.plot(t, y_test, 'b', linewidth=2, label='真实值')
# plt.plot(t, y_pred, 'y', linewidth=2, label='预测值')
# plt.title('线性回归之时间与耗电量之间的关系', fontsize=20)
# plt.xlabel('时间')
# plt.ylabel('耗电量')
# plt.legend(loc='upper left')
# plt.grid(b=True)

fig_degree, axes_degree = plt.subplots(2,1)
axes_degree[0].plot(list, error_train)
axes_degree[1].plot(list, error_test)
plt.show()

# ######################################
# 模型保存
# ######################################
# joblib.dump(ss, 'result\househole_appliances_ss')
# joblib.dump(lr, 'result\househole_appliances_lr')

# ######################################
# 模型
# ######################################