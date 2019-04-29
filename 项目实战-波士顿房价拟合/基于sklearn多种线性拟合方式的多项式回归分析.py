import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 防止中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
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

## RidgeCV和Ridge的区别是：前者可以进行交叉验证
models = [
    Pipeline([
        #  include_bias: 是否添加多项式扩展中的1
            ('Poly', PolynomialFeatures(include_bias=True)),
            ('Linear', LinearRegression(fit_intercept=False))
        ]),
    Pipeline([
            ('Poly', PolynomialFeatures(include_bias=True)),
            # alpha给定的是Ridge算法中，L2正则项的权重值，也就是ppt中的兰姆达
            # alphas是给定CV交叉验证过程中，Ridge算法的alpha参数值的取值的范围
            ('Linear', RidgeCV(alphas=np.logspace(-3,2,50), fit_intercept=False))
        ]),
    Pipeline([
            ('Poly', PolynomialFeatures(include_bias=True)),
            ('Linear', LassoCV(alphas=np.logspace(0,1,10), fit_intercept=False))
        ]),
    Pipeline([
            ('Poly', PolynomialFeatures(include_bias=True)),
            # la_ratio：给定EN算法中L1正则项在整个惩罚项中的比例，这里给定的是一个列表；
            # 表示的是在CV交叉验证的过程中，EN算法L1正则项的权重比例的可选值的范围
            ('Linear', ElasticNetCV(alphas=np.logspace(0,1,10), l1_ratio=[.1, .5, .7, .9, .95, 1], fit_intercept=False))
        ])
]

## 线性回归、Lasso回归、Ridge回归、ElasticNet比较
N = 2
plt.figure(facecolor='w')
degree = np.arange(1, N, 2)  # 阶， 多项式扩展允许给定的阶数
dm = degree.size
colors = []  # 颜色
for c in np.linspace(16711680, 255, dm):
    colors.append('#%06x' % int(c))
titles = [u'线性回归', u'Ridge回归', u'Lasso回归', u'ElasticNet']

for t in range(1):
    model = models[t]  # 选择了模型--具体的pipeline(线性、Lasso、Ridge、EN)
    # print(type(model))
    plt.subplot(2, 2, t + 1)  # 选择具体的子图
    plt.plot(x_test, y_test, 'ro', ms=10, zorder=N - 1)  # 在子图中画原始数据点； zorder：图像显示在第几层

    # 遍历不同的多项式的阶，看不同阶的情况下，模型的效果
    for i, d in enumerate(degree):
        # 设置阶数(多项式)
        model.set_params(Poly__degree=d)
        # 模型训练
        model.fit(x_train, y_train)

        # 获取得到具体的算法模型
        # model.get_params()方法返回的其实是一个dict对象，后面的Linear其实是dict对应的key
        # 也是我们在定义Pipeline的时候给定的一个名称值
        lin = model.get_params()['Linear']
        # 打印数据
        # output = u'%s:%d阶，系数为：' % (titles[t], d)
        # 判断lin对象中是否有对应的属性
        # if hasattr(lin, 'alpha_'):  # 判断lin这个模型中是否有alpha_这个属性
        #     idx = output.find(u'系数')
        #     output = output[:idx] + (u'alpha=%.6f, ' % lin.alpha_) + output[idx:]
        # if hasattr(lin, 'l1_ratio_'):  # 判断lin这个模型中是否有l1_ratio_这个属性
        #     idx = output.find(u'系数')
        #     output = output[:idx] + (u'l1_ratio=%.6f, ' % lin.l1_ratio_) + output[idx:]
        # line.coef_：获取线性模型的参数列表，也就是我们ppt中的theta值，ravel()将结果转换为1维数据
        # print(output, lin.coef_.ravel())

        # 数据预测
        # print('转换前,x_test是:', x_test.shape)
        # x_test = model.fit_transform(x_test)
        y_hat = model.predict(x_test)
        # 计算准确率
        # s = model.score(y_test, y_hat)

        # 当d等于5的时候，设置为N-1层，其它设置0层；将d=5的这条线凸显出来
        z = N + 1 if (d == 5) else 0
        # label = u'%d阶, 正确率=%.3f' % (d, s)
        plt.plot(x_test, y_hat, color=colors[i], lw=2, alpha=0.75, zorder=z)

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title(titles[t])
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
plt.tight_layout(1, rect=(0, 0, 1, 0.95))
plt.suptitle(u'各种不同线性回归过拟合显示', fontsize=22)
plt.show()
plt.savefig('./result/boston.png')











