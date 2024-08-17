# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:26:53 2024

@author: Administrator
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from core import Variable
from core import Add
from core import Mul



def test_Variable():
  data = np.array(1.0)
  x = Variable(data)
  print(x.data)
  data = np.array([1, 2, 3])
  x.data = data
  
def test_Add():
  f = Add()
  a = Variable(np.array(1))
  b = Variable(np.array(2))
  result = f(a, b)
  assert result.data == 3
  result.backward()
  assert a.grad == 1

def test_Mul():
  f = Mul()
  a = Variable(np.array(2))
  b = Variable(np.array(4))
  result = f(a, b)
  assert result.data == 8
  result.backward()
  assert a.grad == 4
  assert b.grad == 2