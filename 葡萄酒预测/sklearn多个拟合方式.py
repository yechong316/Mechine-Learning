import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

## 拦截异常
warnings.filterwarnings(action = 'ignore')
# warnings.filterwarnings(action = 'ignore', category=UserWarning)

# 数据加载
data = pd.read_csv(r"D:\旧电脑\人工智能\05_随堂代码\[20181104_1110_1111]_回归算法\datas\winequality-white.csv")
# print(data.info())
## 自变量名称
names = ["fixed acidity","volatile acidity","citric acid",
         "residual sugar","chlorides","free sulfur dioxide",
         "total sulfur dioxide","density","pH","sulphates",
         "alcohol", "type"]

## 因变量名称
quality = "quality"
# print(data.info())
X = data.iloc[:, 0:-1]
# print(X)/
Y = data[quality]
# # print(x.head())
# print(y)
# 数据清洗

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y)

# 数据训练
N = 5
regression = ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet']
t = np.arange(len(x_test))
plt.figure(figsize=(16,8), facecolor='w')
d_pool = np.arange(1,4,1) # 1 2 3 阶
m = len(d_pool)
clrs = ['#%06x' % int(c) for c in np.linspace(5570560, 255, m + 1)] # 颜色
models = (
    Pipeline(steps=[
        ('Poly', PolynomialFeatures()),
        ('algo', LinearRegression())
    ]),
    Pipeline(steps=[
        ('Poly', PolynomialFeatures()),
        ('algo', LassoCV(alphas=np.logspace(-4, 2, 20)))
    ]),
    Pipeline(steps=[
        ('Poly', PolynomialFeatures()),
        ('algo', RidgeCV(alphas=np.logspace(-4, 2, 20)))
    ]),
    Pipeline(steps=[
        ('Poly', PolynomialFeatures()),
        ('algo', ElasticNetCV(alphas=np.logspace(-4, 2, 20), l1_ratio=np.logspace(0, 1, 5)))
    ]),
)

for j in range(4):

    model = models[j]
    plt.subplot(2, 2, j + 1)
    plt.plot(t, y_test, c='r', lw=2, alpha=0.75, zorder=10, label=u'真实值')

    for i in range(1, N-1):  #不同阶次

        model.set_params(Poly__degree=i)
        # x_train = pipeline.fit_transform(x_train)
        model.fit(x_train, y_train)

        # 数据结果
        # print("阶次为{},各个特征属性的权重系数，也就是ppt上的theta值:{}".format(i, algo.get_params()['algo'].coef_))
        # print("阶次为{},截距项值:{}".format(i, algo.steps[-1][1].intercept_))
        # b. 直接通过评估相关的API查看效果
        lin = model.get_params()['algo']
        # print(type(lin))
        R2_train, R2_test = model.score(x_train, y_train),model.score(x_test, y_test)
        result= u"回归方式:{},阶次:{},训练集R2:{}, 测试集{}".format(regression[j],i,R2_train, R2_test )
        print(result)
        # 在测试的时候对特征属性数据必须做和训练数据完全一样的操作
        # print("当采用{},阶次为{},模型在测试数据上的效果(R2)：{}".format(regression[j],i, ))
        y_pre = model.predict(x_test)
        plt.plot(t, y_pre, c=clrs[i-1], lw=2, alpha=0.75, zorder=10, label=result)
    plt.legend(loc='upper left')
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.grid(True)

plt.suptitle(u'葡萄酒质量预测', fontsize=22)
plt.show()

# 数据展示

# 数据保存