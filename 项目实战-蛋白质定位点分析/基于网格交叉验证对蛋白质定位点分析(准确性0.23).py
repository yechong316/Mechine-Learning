'''

http://archive.ics.uci.edu/ml/datasets/Ecoli
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

def R2_score(y_true, y_pred):
    assert y_pred.shape == y_true.shape, '输入的{}与{}不相等'.format(y_pred.shape, y_true.shape)
    return np.sum(np.square(y_pred - y_true)) / np.sum(np.square(y_pred - np.mean(y_true)))

# 数据加载
names = ['mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'classtic']
df = pd.read_csv('./datas/ecoli.data', sep='  ', names=names)
le = LabelEncoder()
X = df.iloc[:, 0:-1]
Y = pd.DataFrame(le.fit_transform(df.iloc[:, -1]))

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y)

x_train = np.mat(x_train)
y_train = np.mat(y_train).reshape(-1, 1)

# 特征工程
pipeline = Pipeline(steps=[
    ('poly', PolynomialFeatures()),  # 指定第一步做什么操作
    ('algo', ElasticNet(random_state=0))  # 指定最后一步做什么操作，最后一步一般为模型对象
])

params = {
    "poly__degree": [1, 2, 3, 4, 5],
    "algo__alpha": [0.1, 0.01, 1.0, 10.0, 100.0, 1000.0],
    "algo__l1_ratio": [0.1, 0.3, 0.5, 0.9, 0.95, 1.0],
    "algo__fit_intercept": [True, False]
}

algo = GridSearchCV(estimator=pipeline, cv=3, param_grid=params)

# 6. 模型的训练
algo.fit(x_train, y_train)



# 7. 模型效果评估
print("最优参数:{}".format(algo.best_params_))
print("最优参数对应的最优模型:{}".format(algo.best_estimator_))
print("最优模型对应的这个评估值:{}".format(algo.best_score_))

best_pipeline = algo.best_estimator_
best_lr = best_pipeline.get_params()['algo']
# print("各个特征属性的权重系数，也就是ppt上的theta值:{}".format(best_lr.coef_))
print("截距项值:{}".format(best_lr.intercept_))

pred_train = algo.predict(x_train)
pred_test = algo.predict(x_test)
# b. 直接通过评估相关的API查看效果
print("模型在训练数据上的效果(R2)：{}".format(r2_score(pred_train, y_train)))
print("模型在测试数据上的效果(R2)：{}".format(r2_score(pred_test, y_test)))

# 模型展示
t = np.arange(len(x_test))
plt.plot(t, y_test, 'r', label=u'实际值')
plt.plot(t, pred_test, 'b', label=u'预估值')
plt.legend(loc='lower right')
plt.show()
