import numpy as np


def binary_logistic_regression(data, label, lr=0.1, epoch=1):
    assert len(data.shape) == 2
    assert len(label.shape) == 1
    assert data.shape[0] == label.shape[0]

    feature_size = data.shape[-1]

    # initalize trainable variables
    weight = np.random.rand(feature_size, 1).astype(np.float32)
    bias = np.random.rand(1).astype(np.float32)

    for _ in range(epoch):
        pred = go_forward(data, weight, bias)
        weight, bias = go_backward(weight, bias, pred, label, data, lr)
    return pred


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


def go_forward(data, weight, bias):
    z = data @ weight + bias
    return sigmoid(z)


def go_backward(weight, bias, y_pred, y_true, data, lr):
    # update weight
    w_grad = (y_pred - y_true) @ data
    weight -= lr * w_grad
    # update bias
    b_grad = np.mean(y_pred - y_true)
    bias -= lr * b_grad
    return weight, bias


def cross_entropy(y_pred, y_true):
    return - (y_true * np.log(y_pred)) - ((1 - y_true) * np.log(1 - y_pred))


def to_flatten_int(pred):
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    pred = np.transpose(pred).astype(np.int32)
    return pred
