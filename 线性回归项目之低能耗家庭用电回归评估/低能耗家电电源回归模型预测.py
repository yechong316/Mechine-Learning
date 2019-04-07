import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
# from time2str import *

from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
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
print(X.head()
      )
# ######################################
# 数据分割
# ######################################
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75)


# ######################################
# 构造特征工程
# ######################################
ss = StandardScaler()
print(x_train)
X_train = ss.fit_transform(x_train)


# ######################################
# 开始训练
# ######################################
lr = LinearRegression(fit_intercept=False)
lr.fit(X_train, y_train)
y_pred = lr.predict(x_test)

print('训练集上误差为:', lr.score(X_train, y_train))
print('测试集上误差为:', lr.score(x_test, y_test))


# ######################################
# 模型展示
# ######################################
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'b', linewidth=2, label='真实值')
plt.plot(t, y_pred, 'y', linewidth=2, label='预测值')
plt.title('线性回归之时间与耗电量之间的关系', fontsize=20)
plt.xlabel('时间')
plt.ylabel('耗电量')
plt.legend(loc='upper left')
plt.grid(b=True)
# plt.show()

# ######################################
# 模型保存
# ######################################
joblib.dump(ss, 'result\househole_appliances_ss')
joblib.dump(lr, 'result\househole_appliances_lr')

# ######################################
# 模型
# ######################################