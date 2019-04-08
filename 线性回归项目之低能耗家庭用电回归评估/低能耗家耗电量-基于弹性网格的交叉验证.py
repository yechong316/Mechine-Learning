import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


from sklearn.model_selection import  train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV



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
X = df.iloc[:, 2:6]
Y = df['Appliances']
# print(X.head())
# ######################################
# 数据分割
# ######################################
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75)
# print('x_train:', x_train.head(3))
# print('x_test:', x_test.info())

# ######################################
# 多项式拟合
# ######################################
rank = [3, 5, 7, 9]
y_pre = []

# ###############################
# 构建网格参数
# ###############################
pipe = Pipeline(steps=[
    ('poly', PolynomialFeatures()),
    ('algo',ElasticNet(random_state=2))
])

params = {
    "poly__degree": [1, 2, 3,4],
    'algo__alphs': [.1, .2, 3,10],
    'algo__l1_ratio': [0.1, 0.3, 0.5, 0.9, 0.95, 1.0],
    'algo__fit_intercept': [True, False]
}
algo = GridSearchCV(estimator=pipe, cv=3, param_grid=params)

# ###############################
# 模型训练
# ###############################
algo.fit(x_train, y_train)


print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(algo.coef_))
print('训练集上误差为:', algo.score(x_train, y_train))
print('测试集上误差为:', algo.score(x_test, y_test))


# fig, ax = plt.subplot(len(rank),1)
# t = np.arange(len(x_test))
# for i in range(len(rank)):
#     ax[i].plot(t, y_test, 'w', linewidth=2)
#     ax[i].plot(t, y_pre[i], 'w', linewidth=2)
#
#
# plt.show()