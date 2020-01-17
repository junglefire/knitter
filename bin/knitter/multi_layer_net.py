#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import OrderedDict
import numpy as np
import sys
import os

from derivative import numerical_gradient as num_grad
from layers import *

## 多层神经网络
## activation : 'relu' or 'sigmoid'
class MultiLayerNet:
	def __init__(self, input_size, hidden_size_list, output_size, activation = 'relu', weight_init_std = 'relu', weight_decay_lambda = 0):
		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size_list = hidden_size_list
		self.hidden_layer_num = len(hidden_size_list)
		self.weight_decay_lambda = weight_decay_lambda
		# 存储权重和偏置
		self.params = {}
		# 存储层 
		self.layers = OrderedDict()
		# 初始化权重
		self.__init_weight(weight_init_std)
		# 设置激活函数
		activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
		# 创建多层神经网络 
		for idx in range(1, self.hidden_layer_num+1):
			self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
			self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
		# 创建输出层
		idx = self.hidden_layer_num + 1
		self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
		# 输出层的激活函数使用Softmax-With-Loss
		self.last_layer = SoftmaxWithLoss()

	def __init_weight(self, weight_init_std):
		all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
		for idx in range(1, len(all_size_list)):
			scale = weight_init_std
			# Xavier初始值是以激活函数是线性函数为前提而推导出来的。因为sigmoid函数和tanh 
			# 函数左右对称，且中央附近可以视作线性函数，所以适合使用`Xavier初始值`
			# 当激活函数使用ReLU时，一般推荐使用ReLU专用的初始值，也就是`He初始值`
			if str(weight_init_std).lower() in ('relu', 'he'):
				scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLU权重初始值最优计算方法
			elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
				scale = np.sqrt(1.0 / all_size_list[idx - 1])  # sigmoid权重初始值最优计算方法
			# 设定权重
			self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
			self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

	def predict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)
		return x

	def loss(self, x, t):
		y = self.predict(x)
		weight_decay = 0
		for idx in range(1, self.hidden_layer_num + 2):
			W = self.params['W' + str(idx)]
			weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
		return self.last_layer.forward(y, t) + weight_decay

	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		if t.ndim != 1: 
			t = np.argmax(t, axis=1)
		accuracy = np.sum(y == t) / float(x.shape[0])
		return accuracy

	def numerical_gradient(self, x, t):
		loss_W = lambda W: self.loss(x, t)
		grads = {}
		for idx in range(1, self.hidden_layer_num+2):
			grads['W' + str(idx)] = num_grad(loss_W, self.params['W' + str(idx)])
			grads['b' + str(idx)] = num_grad(loss_W, self.params['b' + str(idx)])
		return grads

	def gradient(self, x, t):
		# forward
		self.loss(x, t)
		# backward
		dout = 1
		dout = self.last_layer.backward(dout)
		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)
		# 設定
		grads = {}
		for idx in range(1, self.hidden_layer_num+2):
			grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
			grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
		return grads


## 测试多层神经网络
if __name__== "__main__":
	mn = MultiLayerNet(input_size = 16, hidden_size_list = [10, 10, 10], output_size = 8)
	print(mn.layers)


