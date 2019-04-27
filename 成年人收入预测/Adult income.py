import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp

from sklearn.model_selection import train_test_split


# 数据加载

names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
         'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income']

df = pd.read_csv('./Adult/adult_1.data', sep=',', header=None, names=names)
# print(df.info())
print(df.head())