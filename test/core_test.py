# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:26:53 2024

@author: Administrator
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from plum import Variable


def test_Variable():
  data = np.array(1.0)
  x = Variable(data)
  print(x.data)
  data = np.array([1, 2, 3])
  x.data = data
  
def test_Add():
  a = Variable(np.array(1))
  b = Variable(np.array(2))
  result = a + b
  assert result.data == 3
  result.backward()
  assert a.grad == 1

def test_Mul():
  a = Variable(np.array(2))
  b = Variable(np.array(4))
  result = a * b
  assert result.data == 8
  result.backward()
  assert a.grad == 4
  assert b.grad == 2

def test_Sub():
  a = Variable(np.array(10.0))
  b = Variable(np.array(4.5))
  
  result = a - b
  assert result.data == 5.5
  result.backward()
  assert a.grad == 1
  assert b.grad == -1

def test_Div():
  a = Variable(np.array(12))
  b = Variable(np.array(3))
  result = a / b
  assert result.data == 4
  result.backward()
  assert a.grad == 1/3
  assert b.grad == -12/9

def test_pow():
  a = Variable(np.array(2))
  b = 3

  result = a ** b
  assert result.data == 8
  result.backward()

  assert a.grad == 12
