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
  def __init__(self, data, name=None):
    if data is not None:
      if not isinstance(data, array_types):
        raise TypeError('{} is not supported'.format(type(data)))
    self.data = data
    self.name = name
    self.grad = None
    self.creator = None
    self.generation = 0

  def backward(self, retain_grad=False):
    if self.grad is None:
      self.grad = np.ones_like(self.data)

    funcs = []
    seen_set = set()
    def add_func(f):
      if f not in seen_set:
        funcs.append(f)
        seen_set.add(f)
        funcs.sort(key = lambda x: x.generation)

    add_func(self.creator)
    while funcs:
      f = funcs.pop()
      gys = [output().grad for output in f.outputs]
      gxs = f.backward(*gys)
      if not isinstance(gxs, tuple):
        gxs = (gxs, )

      for x, gx in zip(f.inputs, gxs):
        if x.grad is None:
          x.grad = gx
        else:
          x.grad = x.grad + gx

        if x.creator is not None:
          add_func(x.creator)
      if not retain_grad:
        for output in f.outputs:
          output().grad = None


  def set_creator(self, creator):
    self.creator = creator
    self.generation = creator.generation + 1

  @property
  def shape(self):
    return self.data.shape

  @property
  def ndim(self):
    return self.data.ndim

  @property
  def size(self):
    return self.data.size
    
  @property
  def dtype(self):
    return self.data.dtype

  def __len__(self):
    return len(self.data)

  def __repr__(self):
    if self.data is None:
      return 'Variable(None)'
    p = str(self.data).replace('\n', '\n' + ' ' * 9)
    return 'Variable(' + p + ')'
  
def as_variable(obj):
  if isinstance(obj, Variable):
    return obj
  return Variable(obj)

def as_array(x):
  if np.isscalar(x):
    return np.array(x)
  return x

class Function:
  def __call__(self, *inputs):
    inputs = [as_variable(x) for x in inputs]
    xs = [x.data for x in inputs]
    ys = self.forward(*xs)
    if not isinstance(ys, tuple):
      ys = (ys, )
    outputs = [Variable(as_array(y)) for y in ys]

    if Config.enable_backprop:
      self.generation = max([x.generation for x in inputs])
      for output in outputs:
        output.set_creator(self)
      self.inputs = inputs
      self.outputs = [weakref.ref(output) for output in outputs]
    return outputs if len(outputs) > 1 else outputs[0]
  
  def forward(self, xs):
    raise NotImplementedError()
    
  def backward(self, gys):
    raise NotImplementedError()


class Add(Function):
  def forward(self, x0, x1):
    self.x0_shape = x0.shape
    self.x1_shape = x1.shape
    y = x0 + x1
    return y

  def backward(self, gy):
    return gy, gy

def add(x0, x1):
  return Add()(x0, x1)

if __name__ == '__main__':
  data = np.array(1.0)
  x = Variable(data)
  print(x.data)
  data = np.array([1, 2, 3])
  x.data = data
  print(x)
  print(getattr(Config, 'train'))
  with test_mode():
    print('train:', getattr(Config, 'train'))
  print('train:', getattr(Config, 'train'))
  
  f = Add()
  a = Variable(np.array(1))
  b = Variable(np.array(2))
  result = f(a, b)
  assert result.data == 3
  result.backward()
  print(a.grad)
  print(a)