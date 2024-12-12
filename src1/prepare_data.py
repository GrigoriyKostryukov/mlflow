import numpy as np
from sklearn import datasets, svm


def get_iris_dataset():
    iris = datasets.load_iris()
    return iris

def get_linreg_dataset():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    return X, y
