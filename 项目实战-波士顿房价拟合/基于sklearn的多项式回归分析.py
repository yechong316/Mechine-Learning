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
from sklearn.pipeline import Pipeline
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
model = Pipeline([
    ('Poly', PolynomialFeatures()),  # 给定进行多项式扩展操作， 第一个操作：多项式扩展
    ('Linear', LinearRegression(fit_intercept=False))  # 第二个操作，线性回归
])

# 模型训练
t = np.arange(len(x_test))
N = 5
d_pool = np.arange(1, N, 1)  # 阶
m = d_pool.size
clrs = []  # 颜色
for c in np.linspace(16711680, 255, m):
    clrs.append('#%06x' % int(c))
line_width = 3
lr = LinearRegression()
for i, d in enumerate(d_pool):
    plt.subplot(N - 1, 1, i + 1)
    plt.plot(t, y_test, '-', label=u'真实值', ms=10, zorder=N)
    ### 设置管道对象中的参数值，Poly是在管道对象中定义的操作名称， 后面跟参数名称；中间是两个下划线
    model.set_params(Poly__degree=d)  ## 设置多项式的阶乘
    model.fit(x_train, y_train)  # 模型训练
    # Linear是管道中定义的操作名称
    # 获取线性回归算法模型对象
    lin = model.get_params()['Linear']
    output = u'%d阶，系数为：' % d
    print(output, lin.coef_.ravel())

    # 模型结果预测
    y_hat = model.predict(x_test)
    # 计算评估值
    s = model.score(x_test, y_test)

    # 画图
    z = N - 1 if (d == 2) else 0
    label = u'%d阶, R2=%.3f' % (d, s)
    plt.plot(t, y_hat, color=clrs[i], lw=line_width, alpha=0.75, label=label, zorder=z)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylabel(u'%d阶结果' % d, fontsize=12)

## 预测值和实际值画图比较
plt.suptitle(u"线性回归预测时间和功率之间的多项式关系", fontsize=20)
plt.grid(b=True)
plt.show()

# 模型持久化
from sklearn.externals import joblib
joblib.dump(model.get_params()['Poly'], 'scaler.pkl')
joblib.dump(model.get_params()['Linear'], 'algo.pkl')