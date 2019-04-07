from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl


# mpl.rcParams['font.sens-serif'] = [u'senHei']
mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False

# ######################################
# 加载计算好的数据
# ######################################
s = joblib.load('result\househole_appliances_ss')
lr = joblib.load('result\househole_appliances_lr')
data1 = [[17,17,40]]
data2 = s.transform(data1)
# print(data2)
print(lr.predict(data2))


