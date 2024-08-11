# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:37:22 2024

@author: Administrator
"""

import weakref
import contextlib
import numpy as np


class Config:
  enable_backprop = True
  train = True


@contextlib.contextmanager
def using_config(name, value):
  old_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, old_value)
    
  
def no_grad():
  return using_config('enable_backprop', False)

def test_mode():
  return using_config('train', False)
    

try:
  import cupy
  array_types = (np.ndarray, cupy.ndarray)
except ImportError:
  array_types = (np.ndarray)


class Variable:
  def __init__(self, data):
    self.data = data
    
  
def as_variable(obj):
  if isinstance(obj, Variable):
    return obj
  return Variable(obj)
    
class Function:
  def __call__(self, *inputs):
    inputs = [as_variable(x) for x in inputs]
    x = input.data
    y = x ** 2
    output = Variable(y)
    return output
  
  def forward(self, x):
    raise NotImplementedError()
    
  def backward(self, gys):
    raise NotImplementedError()
    
    
if __name__ == '__main__':
  data = np.array(1.0)
  x = Variable(data)
  print(x.data)
  data = np.array([1, 2, 3])
  x.data = data
  print(x.data)
  print(getattr(Config, 'train'))
  with test_mode():
    print('train:', getattr(Config, 'train'))
  print('train:', getattr(Config, 'train'))