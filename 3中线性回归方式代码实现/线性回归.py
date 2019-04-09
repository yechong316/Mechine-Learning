import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, discriminant_analysis, cross_validation

def load_data():
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data,
           diabetes.target,test_size=0.25, random_state=28)
