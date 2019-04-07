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
algo = Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=4)),  # 指定第一步做什么操作
    ('algo', LinearRegression(fit_intercept=True))  # 指定最后一步做什么操作，最后一步一般为模型对象
])
algo.fit(x_train, y_train)


# 7. 模型效果评估
print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(algo.get_params()['algo'].coef_))
print("截距项值:{}".format(algo.steps[-1][1].intercept_))
# b. 直接通过评估相关的API查看效果
print("模型在训练数据上的效果(R2)：{}".format(algo.score(x_train, y_train)))
# 在测试的时候对特征属性数据必须做和训练数据完全一样的操作
print("模型在测试数据上的效果(R2)：{}".format(algo.score(x_test, y_test)))


