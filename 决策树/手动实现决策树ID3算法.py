import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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
    ent = np.sum([-i * np.log2(i) for i in p])

    # print('信息熵为:{}'.format(ent))
    return ent

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
    ent = entropy_master - np.sum(i * j for i, j in list)

    # print('条件信息熵为:{}'.format(ent))
    return ent

# 条件信息熵
def node_conditional_gain_entropy(x_train, y_train, feature, name):

    # 获取当前数据的行号
    index_name = x_train[x_train[name] == feature].index.tolist()
    category = [i[1] for i in Counter(y_train[index_name]).most_common()]
    return entropy(category)

# 特征-属性字典
def dictionary_feature():
    dict = {}
    dict_feature = {}
    '''
    思路: 创建一个字典,字典的keys为每个特征,每个key对应的value是列表,列表中每个元素是元组,元祖中
        均有两个元素,分别为属性以及该属性的个数,
    实现方法:for语句遍历所有的特征, 利用Counter的函数输出每个特征下面的属性以及个数,并且由大到小排列
    eg: {'敲声': [('浊响', 10), ('沉闷', 5), ('清脆', 2)], '纹理': [('清晰', 9), ('稍糊', 5), ('模糊', 3)]....}
    '''
    dict.update({ column:Counter(root_data[column]).most_common() for column in names_X })
    '''
    思路: 创建一个字典,字典的keys为每个特征,每个key对应的value是列表,列表中每个元素是属性
    实现方法:for语句遍历所有的特征, 利用刚刚创建好的字典查询每个特征下面的属性种类,记住
    属性的位置是第一个,所以索引号是0
    eg: {'敲声': ['浊响', '沉闷','清脆'], '纹理': ['清晰', '稍糊', '模糊']....}
    '''
    dict_feature.update({column: [i[0] for i in dict[column]] for column in names_X })
    return dict, dict_feature

# 在当前样本下,信息熵
def current_entropy(y):
    '''
    在当前样本下,信息熵
    :param y:当前样本的类别,
    :return: 当前样本的信息熵
    '''
    Ent = entropy([i[1] for i in Counter(y).most_common()])

    # print('当前样本下,信息熵为{}'.format(Ent))
    return Ent

# 根据X,Y 进行拟合数据
def fit(data):

    X = data.iloc[:, 0:-1]
    Y = data.iloc[:, -1]
    print('X的格式:{},Y的格式:{}'.format(X.shape, Y.shape))
    assert X.shape[0] == Y.shape[0], 'X和Y的样本数应该相等!'
    
    feature = [[i[0] for i in Counter(X[i]).most_common()] for i in names_X]
    Gain_slaves = {}
    
    # 遍历样本的所有特征
    for i in range(len(feature)):
    
        current_feature = names_X[i]
        classtic_slave = [i[1] for i in Counter(X[current_feature]).most_common()]
        ent = [node_conditional_gain_entropy(X, Y, feature[i][j], current_feature) for j in  range(len(feature[i]))]
        Gain_slave = gain(current_entropy(Y), ent, classtic_slave)
        # print('特征:{},条件信息熵:{:.3f}'.format(names_X[i], Gain_slave))
        Gain_slaves.update({Gain_slave:names_X[i]})

    target_feature = [Gain_slaves[k] for k in sorted(Gain_slaves)][-1]

    print('分类特征为:{}'.format(target_feature))
    return target_feature

# 遍历所有特征,得到决策树
def get_set(features):

    print('分类特征为{}'.format(features))

    '''
    思路: 遍历当前节点上的所有分类特征,在每个特征上面,根据该特征下不同的属性值,分成不同的数据块
    root_data[j] == i]  j 是索引该特征列上的所有数据, i 是属性值,返回一个True, False
    root_data[root_data[j] == i]  前面加一个root_data就是让根据布尔值进行分开,最后不同的数据块
    分别储存到不同的元素,拼接成一个列表

    '''
    data_list = [ [root_data[root_data[j] == i] for i in dict_feature[j]] for j in features ]

    for i in range(len(data_list[0])):
        print('第{}个子集的信息为:\n{}\n*********'.format(i+1, data_list[0][i].head()))
    
    return data_list

# 得到当前节点下面的分类特征
def get_node():
    pass

    return feature #返回当前分类依据,即特征

if __name__ == '__main__':
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
    names = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好坏']
    names_X = [names[i] for i in range(len(names) - 1)]
    root_data = pd.DataFrame(dataSet, columns=names)
    dict, dict_feature = dictionary_feature()

    # 开始第一次分支
    split_j = [fit(i) for i in [root_data]]
    # print('开始第一次分支,分类特征为:{}'.format(split_feature))
    df = get_set(split_j)
    # print('开始第一次分支,数据为:{}'.format(df))


    # 开始第二次分支
    split_feature = [fit(i) for i in df]
    print('开始第二次分支,分类特征为:{}'.format(split_feature))
    df = get_set(split_feature)
    print('开始第二次分支,数据为:{}'.format(df))

    # 开始第二次分支
    # df = [get_set(i) for i in split_feature]
    # split_feature = fit(df)


    # while df
    #     df = [get_set(i) for i in split_feature]
    #     split_feature = fit(df)