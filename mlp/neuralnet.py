import numpy as np
from utils import *


class NeuralNetMLP:
    def __init__(self, input_features, hidden_size, classes_size, random_seed=123):
        super().__init__()

        # 多层感知机输入，隐藏、输出层的 size
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.classes_size = classes_size

        # 设置随机种子及生成器
        self.random_seed = random_seed
        generator = np.random.RandomState(self.random_seed)

        # 初始化隐藏层权重和偏置
        self.weight_h = generator.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.input_features))
        self.bias_h = np.zeros(self.hidden_size)

        # 初始化输出层权重和偏置
        self.weight_o = generator.normal(loc=0.0, scale=0.1, size=(self.classes_size, self.hidden_size))
        self.bias_o = np.zeros(self.classes_size)

    def forward(self, x):
        # 隐藏层计算
        # input dim: [mini_batch, input_features]                  x
        #        dot [hidden_features, input_features].T           weight_h.T
        #        add [hidden_features,]                            bias_h
        # output dim: [mini_batch, hidden_features]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # 输出层计算
        # input_dim: [mini_batch, hidden_features]                a_h
        #         dot [output_features, hidden_features].T        weight_o.T
        #         add [output_features,].T                      bias_o
        z_o = np.dot(a_h, self.weight_o.T) + self.bias_o
        a_o = sigmoid(z_o)
        return a_h, a_o

    def backward(self, x, a_h, a_o, y):
        # 梯度下降算法进行权重更新
        # 1. 对标签进行独热编码
        y_onehot = int2onehot(y, self.classes_size)

        # 输出层梯度计算
        # dLoss/dWo = dLoss/dAo * dAo/dZo * dZo/dWo, 其中局部梯度 dZo/dWo
        # DeltaOut =  dLoss/dAo * dAo/dZo
        # 损失函数 loss = mse = (1/n) * (a_o - y_one_hot)^2
        # dim [mini_batch, classes_size]
        d_loss__a_o = 2. * (a_o - y_onehot) / y.shape[0]
        # a_o = 1. / (1. + np.exp(-z_o))  梯度为 a_o*(1.-a_o)
        d_a_o__d_z_o = a_o * (1. - a_o)

        delta_out = d_loss__a_o * d_a_o__d_z_o

        # z_o = a_h * w_o^T + bias_o
        d_z_o__d_w_o = a_h

        # [classes_size, mini_batch] dot [mini_batch, hidden_features]
        # [classes_size, hidden_features]
        d_loss__dw_o = np.dot(delta_out.T, d_z_o__d_w_o)
        # [classes_size, ]
        d_loss__db_o = np.sum(delta_out, axis=0)

        # 隐藏层梯度计算
        # dLoss/dWh = DeltaOut * dZo/dAh * dAh/dZh * dZh/dWh

        # [classes_size, hidden_features]
        d_z_o__a_h = self.weight_o  # z_o = np.dot(a_h, self.weight_o.T) + self.bias_o

        # [mini_batch, classes_size] dot [classes_size, hidden_features] = [mini_batch, hidden_features]
        d_loss__a_h = np.dot(delta_out, d_z_o__a_h)

        # a_h = 1. / (1. + np.exp(-z_h))  梯度为 a_h*(1.-a_h)  [mini_batch, hidden_features]
        d_a_h__d_z_h = a_h * (1. - a_h)

        # [mini_batch, hidden_features] z_h = np.dot(x, self.weight_h.T) + self.bias_h
        d_z_h__d_w_h = x

        # [hidden_features, input_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum(d_loss__a_h * d_a_h__d_z_h, axis=0)

        return d_loss__dw_o, d_loss__db_o, d_loss__d_w_h, d_loss__d_b_h




