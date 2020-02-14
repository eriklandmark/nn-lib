import numpy as np
import random

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_p(z):
    return sigmoid(z) * (1 - sigmoid(z))


h_w = np.array([[0.7015876072998992, 0.7562564828116174],
                [0.4277337936200052, -0.06790825609602713]])
o_w = np.array([[0.5918038437398523, 0.05362806497710171],
                [-0.9854587793773737, 0.3822935148474702]])

h_b = np.array([0.657822152377201, -0.7083939256325933])
o_b = np.array([0.10616165086841844, 0.5870065599277066])

a = np.array([5,4])[np.newaxis]


def train(data, label):
    z1 = h_w.dot(data) + h_b
    a1 = sigmoid(z1)
    z2 = o_w.dot(a1) + o_b
    a2 = sigmoid(z2)
    # print(a2)

    error = label - a2
    errorHidden = o_w.T.dot(error)

    gradient_w_o = sigmoid_p(a2) * error * 0.1
    dw_o = (np.array(gradient_w_o)[np.newaxis]).T.dot([a1])

    gradient_w_h = sigmoid_p(a1) * errorHidden * 0.1
    dw_h = (np.array(gradient_w_h)[np.newaxis]).T.dot([data])
    # print(dw_h)

    o_w + dw_o
    h_w + dw_h
    o_b + gradient_w_o
    h_b + gradient_w_h


data = [
    ([1,0], [1,0]),
    ([0,1], [1,0]),
    ([1,1], [0,1]),
    ([0,0], [0,1])
]


for i in range(100000):
    random.shuffle(data)
    for i in range(4):
        ex, label = data[i]
        train(np.array(ex), np.array(label))


# train(np.array([1, 0]), np.array([1, 0]))

def predict(example):
    z1 = h_w.dot(example) + h_b
    a1 = sigmoid(z1)
    z2 = o_w.dot(a1) + o_b
    a2 = sigmoid(z2)
    print(a2)


predict(np.array([1,0]))