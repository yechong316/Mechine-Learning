'''

http://archive.ics.uci.edu/ml/datasets/Ecoli
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

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
# print(Y)

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y)

# print('训练集的数据格式为{}'.format(x_train.shape))
# print('测试集的数据格式为{}'.format(y_train.shape))
x_train = np.mat(x_train)
y_train = np.mat(y_train).reshape(-1, 1)

# 特征工程
algo = DecisionTreeClassifier(min_impurity_decrease=0.0
)

# 6. 模型的训练
algo.fit(x_train, y_train)



# 7. 模型效果评估

pred_train = algo.predict(x_train)
pred_test = algo.predict(x_test)
print("模型在训练数据上的效果(R2)：{}".format(r2_score(pred_train, y_train)))
print("模型在测试数据上的效果(R2)：{}".format(r2_score(pred_test, y_test)))

# 模型展示
# b. 可视化方式二：直接使用pydotpuls库将数据输出为pdf或者png格式
from sklearn import tree
import pydotplus
# print(pd.unique(Y))
dot_data = tree.export_graphviz(decision_tree=algo, out_file=None,
                                feature_names=['mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2'],
                                class_names=le.classes_,
                                filled=True, rounded=True,
                                special_characters=True
                                )

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('./result/iris0.png')
graph.write_pdf('./result/iris0.pdf')