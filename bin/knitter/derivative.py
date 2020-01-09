#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

## 数值差分
def numerical_diff(f, x):
	h = 1e-4 # 0.0001 
	return (f(x+h) - f(x-h)) / (2*h)

## 梯度
## 参数f为函数, x为NumPy数组，函数对x的各个元素求数值微分
def _numerical_gradient_no_batch(f, x):
	# 0.0001
	h = 1e-4
	# 生成一个形状和x相同、所有元素都为0的数组
	grad = np.zeros_like(x)
	for idx in range(x.size):
		tmp_val = x[idx]
		# f(x+h)的计算
		x[idx] = float(tmp_val) + h
		fxh1 = f(x)
		# f(x-h)的计算
		x[idx] = tmp_val - h 
		fxh2 = f(x) 
		grad[idx] = (fxh1 - fxh2) / (2*h)
		x[idx] = tmp_val
	return grad

## 梯度
def numerical_gradient(f, X):
	if X.ndim == 1:
		return _numerical_gradient_no_batch(f, X)
	else:
		grad = np.zeros_like(X)
		for idx, x in enumerate(X):
			grad[idx] = _numerical_gradient_no_batch(f, x)
		return grad

## 梯度下降
def gradient_descent(f, init_x, lr=0.01, step_num=100):
	x = init_x
	x_history = []
	for i in range(step_num):
		x_history.append(x.copy())
		grad = numerical_gradient(f, x)
		x -= lr * grad
	return x, np.array(x_history)
