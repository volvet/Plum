# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 16:47:14 2024

@author: Administrator
"""

import numpy as np


def sum_to(x, shape):
  ndim = len(shape)
  lead = x.ndim - ndim
  lead_axis = tuple(range(lead))
  axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
  #print(axis)
  #print(lead_axis)
  #print(axis + lead_axis)
  y = x.sum(lead_axis + axis, keepdims=True)
  if lead > 0:
    y = y.squeeze(lead_axis)
  return y


if __name__ == '__main__':
  a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  y = sum_to(a, [1])
  print(y)
