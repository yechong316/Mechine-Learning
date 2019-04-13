import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter


# 信息熵
def entropy(category):
    '''
    求信息熵
    :param category: 输入长度类别种数N, 各个元素为各种类别中的数量的列表
    :return:
    '''
    # if 0 == category:
    #     return 0
    p = [i / np.sum(category) for i in category]
    return np.sum([-i * np.log2(i) for i in p])

# 信息增益
def gain(entropy_master, entropy_slave, classtic_slave):
    '''
    传入样本信息熵,样本条件信息熵,以及类别数量,得到信息增益熵
    :param entropy_master: float, 样本信息熵
    :param entropy_slave: list, 样本条件信息熵
    :param classtic: list, 类别数量
    :return: 信息增益熵
    '''
    p = [i / np.sum(classtic_slave) for i in classtic_slave]

    list = zip(entropy_slave, p)
    return entropy_master - np.sum(i * j for i, j in list)

dataSet = [
    # 1
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    # 2
    ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
    # 3
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    # 4
    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
    # 5
    ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    # 6
    ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
    # 7
    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
    # 8
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],

    # ----------------------------------------------------
    # 9
    ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
    # 10
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
    # 11
    ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
    # 12
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
    # 13
    ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
    # 14
    ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
    # 15
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
    # 16
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
    # 17
    ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
]
names = ['色泽', '根蒂', '敲声', '纹理' ,'脐部', '触感', '好坏']
names_X = ['色泽', '根蒂', '敲声', '纹理' ,'脐部', '触感']

df = pd.DataFrame(dataSet, columns=names)
'''
1.计算当前样本集合的根节点信息熵 entropy
    (1) 样本有N个类别,就建立含有N个元素的列表
    (2) 统计各个元素的个数,并求概率值,
    (3) 将该列表传入entropy函数中,得到值
2.遍历当前样本的特征,对每个属性求信息增益 gain,并储存到字典中
3.取出gain最大的属性,将其作为分类属性
4.根据该属性的特征值进行分类,有N个特征值,则分为N个子集合i
5.遍历所有子集合,求各个子集合的信息熵
6.针对每个子集合,重复2~4步骤
7.终止条件:
    (1) -属性全部遍历完毕
    (2) -当前属性下,特征值全部一样或为空
    (3) -当前数据全部为一类,无需划分

代码部分:要求统计个数
'''

X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, Y)

# print(Counter(y_train).most_common())
# feature_values =
category_master = [i[1] for i in Counter(y_train).most_common()]

Ent_master = entropy(category_master)
# print(Ent)

feature = [[i[0] for i in Counter(x_train[_]).most_common()] for _ in names_X]
Gain_slave = ['ent_' + str(i) for i in range(len(feature))]
Gain_slaves = []
for i in range(len(feature)):
    # print(feature[i])
    ent = []
    # print('特征为{}时, 属性个数分别为{}'.format(names_X[i],Counter(x_train[names_X[i]]).most_common()))
    classtic_slave = [i[1] for i in Counter(x_train[names_X[i]]).most_common()]
    for j in  range(len(feature[i])):
        index_name = x_train[x_train[names_X[i]] == feature[i][j]].index.tolist()
        # print('特征为:{}, 属性值为{}\n索引到的数据为\n{},\n索引号为:\n{}'.format(feature[i],
        # feature[i][j], x_train[x_train[names_X[i]] == feature[i][j]], index_name))
        # print(y_train[index_name])
        category = [i[1] for i in Counter(y_train[index_name]).most_common()]
        Ent = entropy(category)
        # ent_slave.append(Ent)
        # print(Ent)
        # print('当前分类特征为:{}, 属性值为{},\n信息熵为{}'.format(feature[i],
        # feature[i][j], Ent))
        ent.append(Ent)
        # print('*'*10)

    Gain_slave[i] = gain(Ent_master, ent, classtic_slave)
    # print('各个参数分别为{}, {}, {}'.format(Ent_master, ent, classtic_slave))
    print('特征:{},条件信息熵:{:.3f}'.format(names_X[i], Gain_slave[i]))


#     求该条件下的增益
# print([[i[0] for i in Counter(x_train[_]).most_common()] for _ in names_X])
# for _ in names:
#     print( [i[0] for i in Counter(x_train[_]).most_common()])


# category = [i[1] for i in Counter(x_train).most_common()]


# for i in feature_values[-1]:
#     print(i)
#     pass


# df_slvae1 = df.loc[lambda df:df['色泽'] == '青绿', :]
# print(df_slvae1.drop(labels='色泽', axis=1))




