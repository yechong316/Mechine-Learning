import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
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

# ######################################
# 数据分割
# ######################################
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75)

# 方案三: 管道流
linear_model = ['ElasticNet','Lasso','Ridge','RidgeCV','LinearRegression']


# ElasticNet
algo_ElasticNet = Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=4)),  # 指定第一步做什么操作
    ('algo', ElasticNet(fit_intercept=True))  # 指定最后一步做什么操作，最后一步一般为模型对象
])
algo_ElasticNet.fit(x_train, y_train)
y_ElasticNet = algo_ElasticNet.predict(x_test)
# print(y_ElasticNet)
# Lasso
algo_Lasso = Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=4)),  # 指定第一步做什么操作
    ('algo', Lasso(fit_intercept=True))  # 指定最后一步做什么操作，最后一步一般为模型对象
])
algo_Lasso.fit(x_train, y_train)
y_Lasso = algo_ElasticNet.predict(x_test)

# Ridge
algo_Ridge = Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=4)),  # 指定第一步做什么操作
    ('algo', Ridge(fit_intercept=True))  # 指定最后一步做什么操作，最后一步一般为模型对象
])
algo_Ridge.fit(x_train, y_train)
y_Ridge = algo_ElasticNet.predict(x_test)

# RidgeCV
algo_RidgeCV = Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=4)),  # 指定第一步做什么操作
    ('algo', RidgeCV(fit_intercept=True))  # 指定最后一步做什么操作，最后一步一般为模型对象
])
algo_RidgeCV.fit(x_train, y_train)
y_RidgeCV = algo_ElasticNet.predict(x_test)

# LinearRegression
algo_LinearRegression = Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=4)),  # 指定第一步做什么操作
    ('algo', LinearRegression(fit_intercept=True))  # 指定最后一步做什么操作，最后一步一般为模型对象
])
algo_LinearRegression.fit(x_train, y_train)
y_LinearRegression = algo_ElasticNet.predict(x_test)


algo_total = ['algo_ElasticNet', 'algo_Lasso', 'algo_Ridge', 'algo_RidgeCV', 'algo_LinearRegression']

print(algo_total)
# 7. 模型效果评估
# print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(algo.get_params()['algo'].coef_))
# print("截距项值:{}".format(algo.steps[-1][1].intercept_))
# # b. 直接通过评估相关的API查看效果
# print("模型在训练数据上的效果(R2)：{}".format(algo.score(x_train, y_train)))
# # 在测试的时候对特征属性数据必须做和训练数据完全一样的操作
# print("模型在测试数据上的效果(R2)：{}".format(algo.score(x_test, y_test)))
t = np.arange(len(x_test))
color = ['b',
'g',
'r',
'c',
'm',
'y',
'k']
plt.plot(t, y_ElasticNet,'b' )
plt.plot(t, y_Lasso, 'g')
plt.plot(t, y_LinearRegression, 'r')
plt.plot(t, y_Ridge, 'm')
plt.plot(t, y_RidgeCV, 'y')
plt.xlabel('时间')
plt.ylabel('耗电量')
plt.title('5种线性模型下,预测值与真实值的对比')
plt.grid()
# plt.show()
print(y_ElasticNet == y_Lasso)

