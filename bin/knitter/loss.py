#!/usr/bin/env python
# -*- coding:utf-8 -*-
import scipy.special as ss
import numpy as np

## MSE
def mean_squared_error(y, t):
	return 0.5*np.sum((y-t)**2)

## 批量处理交叉熵
## 注意：标签必须采用one-hot编码
def cross_entropy_error(y, t, onehot = True):
	if y.ndim == 1:
		t = t.reshape(1, t.size) 
		y = y.reshape(1, y.size)
	batch_size = y.shape[0] 
	if onehot:
		return -np.sum(t*np.log(y+1e-7))/batch_size
	else:
		# 如果标签不是one-hot编码，采用下面这个语句
		# `np.arange(batch_size)`会生成一个从0到batch_size-1的数组
		# 比如，当batch_size为5时，`np.arange(batch_size)`会生成一个NumPy数组[0, 1, 2, 3, 4]
		# 因为t中标签是以`[2, 7, 0, 9, 4]`的形式存储的，所以`y[np.arange(batch_size), t]`能抽
		# 出各个数据的正确解标签对应的神经网络的输出
		# 在这个例子中，`y[np.arange(batch_size), t]`会生成NumPy数组
		# `[y[0,2], y[1,7], y[2,0], y[3,9], y[4,4]]`
		return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

## main
if __name__ == "__main__":
	t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
	y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
	y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

	print("MSE : ", mean_squared_error(np.array(y1), np.array(t)))
	print("MSE : ", mean_squared_error(np.array(y2), np.array(t)))
	print("CROSS : ", cross_entropy_error(np.array(y1), np.array(t)))
	print("CROSS : ", cross_entropy_error(np.array(y2), np.array(t)))
	pass