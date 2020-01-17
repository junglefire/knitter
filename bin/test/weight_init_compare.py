#!/usr/bin/env python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.getcwd() + "/bin/knitter")

from multi_layer_net import *
from dataset.mnist import *
from optimizer import *
from util import *

## 加载手写数据集
mnist = Mnist("./mnist")
(x_train, t_train), (x_test, t_test) = mnist.load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

## 初始化权重
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
optimizer = SGD(lr=0.01)
networks = {}
train_loss = {}

for key, weight_type in weight_init_types.items():
	networks[key] = MultiLayerNet(input_size = 784, hidden_size_list = [100, 100, 100, 100],
								  output_size = 10, weight_init_std = weight_type)
	train_loss[key] = []

## 训练模型
for i in range(max_iterations):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]
	
	for key in weight_init_types.keys():
		grads = networks[key].gradient(x_batch, t_batch)
		optimizer.update(networks[key].params, grads)
		loss = networks[key].loss(x_batch, t_batch)
		train_loss[key].append(loss)
	
	if i % 100 == 0:
		print("===========" + "iteration:" + str(i) + "===========")
		for key in weight_init_types.keys():
			loss = networks[key].loss(x_batch, t_batch)
			print(key + ":" + str(loss))

## 绘制图形
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
	plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()
