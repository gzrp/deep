from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from neuralnet import NeuralNetMLP

num_epochs = 100
minibatch_size = 64


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values
y = y.astype(int).values


def data_test():
    print(X.shape)

    print(y.shape)
    print(X[0:3][0:15])
    print(y[0:3])
    print("---")
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X[y == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10000, random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)


# 小批量数据生成器
def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]


def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int2onehot(targets, num_labels)
    return np.mean((onehot_targets - probas) ** 2)


def accuracy(targets, predicted):
    return np.mean(predicted == targets)


def compute_mse_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, total = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predict_labels = np.argmax(probas, axis=1)
        onehot_targets = int2onehot(targets, num_labels)
        loss = np.mean((onehot_targets - probas) ** 2)
        correct_pred += (predict_labels == targets).sum()
        total += targets.shape[0]
        mse += loss
    mse = mse / i
    acc = correct_pred / total
    return mse, acc


def train(model, X_train, y_train, X_valid, y_valid, num_epochs=num_epochs, lr=0.1):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for epoch in range(num_epochs):
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

        for X_train_batch, y_train_batch in minibatch_gen:
            # 前馈计算输出
            a_h, a_o = model.forward(X_train_batch)
            # 反向计算梯度
            d_loss__dw_o, d_loss__db_o, d_loss__d_w_h, d_loss__d_b_h = model.backward(X_train_batch, a_h, a_o, y_train_batch)
            # 更新权重
            model.weight_h -= lr * d_loss__d_w_h
            model.bias_h -= lr * d_loss__d_b_h
            model.weight_o -= lr * d_loss__dw_o
            model.bias_o -= lr * d_loss__db_o

        # 计算损失和准确率, 损失值没有参与反馈过程，仅用来跟踪训练过程
        train_mse, train_acc = compute_mse_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_acc(model, X_valid, y_valid)

        train_acc, valid_acc = train_acc * 100, valid_acc * 100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)

        print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
              f'| Train Loss: {train_mse:.6f} '
              f'| Train Acc: {train_acc:.2f} %'
              f'| Valid Acc: {valid_acc:.2f} %')

    return epoch_loss, epoch_train_acc, epoch_valid_acc




if __name__ == '__main__':
    np.random.seed(123)
    model = NeuralNetMLP(input_features=28*28, hidden_size=50, classes_size=10)
    epoch_loss, epoch_train_acc, epoch_valid_acc = train(model, X_train, y_train, X_valid, y_valid, num_epochs=num_epochs, lr=0.1)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(range(len(epoch_loss)), epoch_loss)
    axes[0].set_ylabel('MSE')
    axes[0].set_xlabel('Epoch')

    axes[1].plot(range(len(epoch_train_acc)), epoch_train_acc, label='Train Acc')
    axes[1].plot(range(len(epoch_valid_acc)), epoch_valid_acc, label='Valid Acc')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='lower right')
    plt.show()

    test_mse, test_acc = compute_mse_acc(model, X_test, y_test)
    print(f'| Test Loss: {test_mse:.6f} '
          f'| Test Acc: {test_acc:.2f} %')

