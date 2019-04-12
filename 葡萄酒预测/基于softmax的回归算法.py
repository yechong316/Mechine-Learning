import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report,confusion_matrix

## 拦截异常
warnings.filterwarnings(action = 'ignore')

# 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']

# 数据加载
data_white = pd.read_csv(r"D:\人工智能\mechine_learning\数据集\葡萄酒.csv")
# data_red = pd.read_csv(r"D:\旧电脑\人工智能\05_随堂代码\[20181104_1110_1111]_回归算法\datas\winequality-red.csv")
# print(data_white.quality)
print(Counter(data_white.quality).most_common())
# data = pd.concat([data_white, data_red], axis=0)
# data
# print("合并后,数据的信息为:\n{}".format(data.head(5)))
X = data_white.iloc[:, 0:-1]
Y = data_white.iloc[:, -1]
# print('X是:\n{}'.format(X))
# print('Y是:\n{}'.format(Y))

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y)
# print('y_train:\n{}'.format(y_train))

# 模型构建
algo = LogisticRegression(multi_class='multinomial',penalty='l2',fit_intercept=True,class_weight=None,n_jobs=4, solver='saga')
algo.fit(x_train, y_train)

# 8. 模型效果评估
train_predict = algo.predict(x_train)
test_predict = algo.predict(x_test)
# print("测试集上的效果(准确率):{}".format(algo.score(x_test, y_test)))
# # print("训练集上的效果(准确率):{}".format(algo.score(x_train, y_train)))
# print("测试集上的效果(准确率):{}".format(accuracy_score(y_test, test_predict)))
# print("训练集上的效果(准确率):{}".format(accuracy_score(y_train, train_predict)))
print("测试集上的效果(结果报告):\n{}".format(classification_report(y_test, test_predict)))
print("训练集上的效果(结果报告):\n{}".format(classification_report(y_train, train_predict)))

# 模型展示
t = np.arange(len(x_test))
plt.figure(figsize=(14, 7), facecolor='w')
plt.ylim(-1, 11)

plt.scatter(t, y_test,  label=u'真实值')
plt.scatter(t, test_predict, label=u'预测值')
plt.xlabel('样本数')
plt.xlabel('葡萄酒质量等级')
plt.title('基于softmax的葡萄酒质量等级回归')
plt.grid()
plt.legend(loc='upper left')
plt.show()