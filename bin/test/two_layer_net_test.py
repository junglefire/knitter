#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.getcwd() + "/bin/knitter")

from two_layer_net import TwoLayerNet
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
learning_rate = 0.1

## 构造两层神经网络
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

## 训练模型
for i in range(iters_num):
	print("... loop#", i)
	# 获取mini-batch 
	batch_mask = np.random.choice(train_size, batch_size) 
	x_batch = x_train[batch_mask] 
	t_batch = t_train[batch_mask]
	# 计算梯度 
	grad = network.numerical_gradient(x_batch, t_batch) 
	## 高速版!
	# grad = network.gradient(x_batch, t_batch) 
	# 更新参数 
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]
	# 记录学习过程 
	loss = network.loss(x_batch, t_batch) 
	train_loss_list.append(loss)

print(train_loss_list)