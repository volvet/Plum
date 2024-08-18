# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:40:30 2024

@author: Administrator
"""

import numpy as np

from core import Function

class Sin(Function):
  def forward(self, x):
    y = np.sin(x)
    return y

  def backward(self, gy):
    x = self.inputs[0].data
    gx = gy * np.cos(x)
    return gx

def sin(x):
  return Sin()(x)


if __name__ == '__main__':
  from core import Variable
  x = Variable(np.array(np.pi/4))
  y = sin(x)
  print(y)
  y.backward()
  print(x.grad)