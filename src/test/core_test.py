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


def test_Variable():
  data = np.array(1.0)
  x = Variable(data)
  print(x.data)
  data = np.array([1, 2, 3])
  x.data = data