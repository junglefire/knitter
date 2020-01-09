#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.getcwd() + "/bin/deep")

from two_layer_net_ex import TwoLayerNetEx
from dataset.mnist import *

## 加载数据集
mnist = Mnist(sys.argv[1])
(x_train, t_train), (x_test, t_test) = mnist.load_mnist(normalize = True, one_hot_label = True)

## 存储每次训练后损失函数的值
train_loss_list = []

## 超参数 
iters_num = 10000 
train_size = x_train.shape[0] 
batch_size = 100 
learning_rate = 0.01
train_loss_list = []
train_acc_list = [] 
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

## 构造两层神经网络
network = TwoLayerNetEx(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
	batch_mask = np.random.choice(train_size, batch_size) 
	x_batch = x_train[batch_mask] 
	t_batch = t_train[batch_mask]
	# 通过误差反向传播法求梯度 
	grad = network.gradient(x_batch, t_batch)
	# 更新 
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]

	loss = network.loss(x_batch, t_batch) 
	train_loss_list.append(loss)

	if i % iter_per_epoch == 0:
		train_acc = network.accuracy(x_train, t_train) 
		test_acc = network.accuracy(x_test, t_test) 
		train_acc_list.append(train_acc) 
		test_acc_list.append(test_acc) 
		print(train_acc, test_acc)

# print(train_loss_list)