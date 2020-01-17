#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.getcwd() + "/bin/knitter")

from multi_layer_net_ex import *
from dataset.mnist import *
from optimizer import *
from trainer import *
from util import *

## 加载手写数据集
mnist = Mnist("./mnist")
(x_train, t_train), (x_test, t_test) = mnist.load_mnist(normalize=True)

x_train = x_train[:300]
t_train = t_train[:300]

## 设置Dropout
use_dropout = False
dropout_ratio = 0.2

network = MultiLayerNetExtend(input_size = 784, hidden_size_list = [100, 100, 100, 100, 100, 100],
                              output_size = 10, use_dropout = use_dropout, dropout_ration = dropout_ratio)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs = 301, mini_batch_size = 100,
                  optimizer = 'sgd', optimizer_param = {'lr': 0.01}, verbose = True)

trainer.train()
train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

## 绘制曲线
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()