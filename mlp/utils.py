import numpy as np

'''
utils 模块中包含激活函数，独热转换等工具函数
'''


def sigmoid(x):
    """
    :param x: numpy array, elem in all R
    :return: sigmoid function apply to x
    """
    y = x.copy()      # 对sigmoid函数优化，避免出现极大的数据溢出
    y[x >= 0] = 1.0 / (1 + np.exp(-x[x >= 0]))
    y[x < 0] = np.exp(x[x < 0]) / (1.0 + np.exp(x[x < 0]))
    return y


def int2onehot(y, num_classes):
    """
    :param y: numpy array, elem in [0~num_classes - 1]
    :param num_classes: label numbers
    :return: onehot numpy array
    """
    ary = np.zeros((y.shape[0], num_classes))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary


__all__ = ['sigmoid', 'int2onehot']
