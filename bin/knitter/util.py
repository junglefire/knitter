#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pprint as pp
import numpy as np

def smooth_curve(x):
	window_len = 11
	s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
	w = np.kaiser(window_len, 2)
	y = np.convolve(w/w.sum(), s, mode='valid')
	return y[5:len(y)-5]


## 优化卷积运算，将卷积核感受野的转化成一行（列）来存储，优化运算速度，减少内存访问时间
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
	N, C, H, W = input_data.shape
	out_h = (H + 2*pad - filter_h)//stride + 1
	out_w = (W + 2*pad - filter_w)//stride + 1

	img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
	col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

	for y in range(filter_h):
		y_max = y + stride*out_h
		for x in range(filter_w):
			x_max = x + stride*out_w
			col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
	# transpose操作很关键，原来shape是(N,C,filter_h,filter_w,out_h,out_w)
	# 现在保持N、out_w、out_h不变，将C、filter_w、filter_h的值放都后面维度，相当于把卷积感受野部分
	# 全放后面维度，然后reshape操作把channel、filter_h、filter_w(卷积感受野部分) 规整成一行，方便
	# 直接与卷积做矩阵乘法
	col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
	return col

## 从矩阵到图像
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
	N, C, H, W = input_shape
	out_h = (H + 2*pad - filter_h)//stride + 1
	out_w = (W + 2*pad - filter_w)//stride + 1
	col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

	img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
	for y in range(filter_h):
		y_max = y + stride*out_h
		for x in range(filter_w):
			x_max = x + stride*out_w
			img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

	return img[:, :, pad:H + pad, pad:W + pad]

