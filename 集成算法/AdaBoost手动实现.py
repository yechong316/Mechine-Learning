import numpy as np
import pysnooper
np.set_printoptions(precision=5)
# 符号函数
def sign(x, thresold=0.0):

    if x <= thresold:
        return 1
    else:
        return -1

def sign_reverse(x, thresold=0.0):

    if x <= thresold:
        return -1
    else:
        return 1

def bool2num(bool_str):
    if bool_str:
        return 0
    else:
        return 1

# 计算分类器系数
def coefficient(x, y, w):
    assert len(x) == len(y)

    error_num = (x == y)

    error = np.sum([
        w_i * bool2num(error_i) for w_i, error_i in zip(w, error_num)
    ])

    return 0.5 * np.log((1 - error) / error)

# 更新权重因子
@pysnooper.snoop('file.log')
def update_weight(w, alpha, y_true, y_pred):

    assert len(w) == len(y_pred) == len(y_true)

    # 逐个取出权重系数, 真实值, 预测值,依次求解规范化因子
    # Z = np.sum([
    #     w_i * np.exp(-alpha * y_i * G_i) for w_i, y_i, G_i in zip(w, y_true, y_pred)
    # ])
    w_new = [
        w_i * np.exp(-alpha * y_i * G_i) for w_i, y_i, G_i in zip(w, y_true, y_pred)
    ]

    total = np.sum(w_new)
    return np.array(w_new) / total

# Adaboost训练
# def Adaboost(thresold, w, y_pred):


# return alpha, w_1, y_pred_new

w = np.ones(10) / 10
X = np.arange(10)
Y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

# print(X)
# print(len(Y))Y

T = 1
thresold = [2.5, 8.5, 5.5]
y_pred_1 = [sign(i, 2.5) for i in X]
alpha_1 = coefficient(y_pred_1, Y, w)
w_1 = update_weight(w, alpha_1, Y, y_pred_1)
x_2 = [alpha_1 * i for i in y_pred_1]
# g_1 =
print('第{}次迭代后,系数是{}, 权重是\n{}\n新的x为{}'.format(T, alpha_1, w_1,x_2))

T = 2
y_pred_2 = [sign(i, 8.5) for i in X]
alpha_2 = coefficient(y_pred_2, Y, w_1)
w_2 = update_weight(w_1, alpha_2, Y, y_pred_2)
x_3 = [alpha_1 * i for i in y_pred_2]
print('第{}次迭代后,系数是{}, 权重是\n{}\n新的x为{}'.format(T, alpha_2, w_2,x_3))


T = 3
y_pred_3 = [sign_reverse(i, 5.5) for i in X]
alpha_3 = coefficient(y_pred_3, Y, w_2)
w_2 = update_weight(w_2, alpha_3, Y, y_pred_3)
x_4 = [alpha_1 * i for i in y_pred_3]
print('第{}次迭代后,系数是{}, 权重是\n{}\n新的x为{}'.format(T, alpha_3, w_2,x_4))

alpha_total = [alpha_1, alpha_2, alpha_3]
y_pred_total = [y_pred_1, y_pred_2, y_pred_3]
# G = alpha_1 * y_pred_1 + alpha_2 * y_pred_2 + alpha_3 * y_pred_3
G = [
    alpha_1 * j_1 + alpha_1 * j_2 + alpha_1 * j_3 for j_1, j_2, j_3 in zip(y_pred_1, y_pred_2, y_pred_3)
]
G_total = [
    sign_reverse(i) for i in G
]
print(G_total == Y)
print(G_total)