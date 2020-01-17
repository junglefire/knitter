#!/usr/bin/env python
# -*- coding:utf-8 -*-
import scipy.special as ss
import numpy as np

## S函数
def sigmoid_function(x):
	return ss.expit(x)

## 阶跃函数
def step_function(x):
	y = x > 0 
	return y.astype(np.int)

## Tanh函数 
def tanh_function(x):
	return np.tanh(x)

## ReLU函数
def relu_function(x):
	return np.maximum(0, x)

## 恒等函数
def identity_function(x):
	return x

## softmax函数的输出是0.0到1.0之间的实数,softmax函数的输出值的总和是1
## 因此,可以把softmax函数的输出解释为`概率`
def softmax_function(x):
	if x.ndim == 2:
		x = x.T
		x = x - np.max(x, axis=0)
		y = np.exp(x) / np.sum(np.exp(x), axis=0)
		return y.T 
	else:
		x = x - np.max(x) # 防溢出対策
		return np.exp(x) / np.sum(np.exp(x))

## main
if __name__ == "__main__":
	import matplotlib.pyplot as plt
	# 显示坐标系
	a = 8
	b = 1.1 
	plt.xlim(-a, a)
	plt.ylim(-b, b)
	# plt.plot([0, 0], [0, a], linewidth = 0.5, color = 'red', linestyle = "--")
	# 绘制sigmoid函数图像
	x = np.arange(-(a-1), (a-1), 0.1)
	y = sigmoid_function(x)
	plt.plot(x, y, color = 'r', label = 'sigmoid')
	# 绘制阶跃函数图像
	y = step_function(x) 
	plt.plot(x, y, color = 'g', label = 'step')
	# 绘制tanh函数图像
	y = tanh_function(x) 
	plt.plot(x, y, color = 'k', label = 'tanh')
	# 绘制ReLU函数图像
	y = relu_function(x) 
	plt.plot(x, y, color = 'c', label = 'ReLU')
	plt.legend()
	plt.show()
