#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Sat  2 Nov 10:05:23 2013 

"""Tests for xbob.core.random
"""

from __future__ import division
from ..random import variate_generator, mt19937, uniform, normal, lognormal
import numpy
import nose.tools

def test_mt19937_creation():

  x = mt19937()
  y = mt19937()
  assert x == y

def test_mt19937_comparison():

  x = mt19937(10)
  y = mt19937(11)
  assert x != y

def test_uniform_creation():

  x = uniform('int8')
  assert x.dtype == numpy.int8
  assert x.min == 0
  assert x.max == 9

def test_uniform_int8():
  
  x = uniform('uint8', min=0, max=7)
  assert x.dtype == numpy.uint8
  rng = mt19937()
  l = [x(rng) for k in range(10000)]
  assert min(l) == 0
  assert max(l) == 7
  assert abs(sum(l)/10000 - 3.5) < 0.1

def test_uniform_float64():

  x = uniform('float64', min=-1, max=+1)
  assert x.dtype == numpy.float64
  rng = mt19937()
  l = [x(rng) for k in range(10000)]
  assert min(l) >= -1.0
  assert max(l) < 1.0
  assert abs(sum(l)/10000) < 0.1

def test_uniform_bool():

  x = uniform(bool)
  assert x.min == False
  assert x.max == True
  rng = mt19937()
  l = [x(rng) for k in range(1000)]
  assert min(l) == False
  assert max(l) == True

@nose.tools.raises(ValueError)
def test_uniform_bool_raises():

  x = uniform(bool, True, True)

@nose.tools.raises(NotImplementedError)
def test_uniform_complex():

  x = uniform('complex64')

def test_mt19937_same_sequence():
  
  x = uniform('float64', min=-1, max=+1)
  rng1 = mt19937(17)
  rng2 = mt19937(17)
  check = [x(rng1) == x(rng2) for k in range(1000)]
  assert numpy.all(check)

def test_mt19937_different_sequences():
  
  x = uniform('float64', min=-1, max=+1)
  rng1 = mt19937(17)
  rng2 = mt19937(-3)
  check = [x(rng1) == x(rng2) for k in range(1000)]
  assert not numpy.all(check)

def test_variate_generator_1d():
  
  import math

  x = variate_generator(mt19937(), uniform('float32', min=0, max=2*math.pi))
  m = x(10)
  assert m.shape == (10,)
  assert m.dtype == numpy.float32

def test_variate_generator_2d():

  x = variate_generator(mt19937(), uniform('uint16', min=0, max=65535))
  m = x((10,10))
  assert m.shape == (10,10)
  assert m.dtype == numpy.uint16

def test_normal():

  x = variate_generator(mt19937(), normal('float64', mean=0.5, sigma=2.0))
  assert x.distribution.mean == 0.5
  assert x.distribution.sigma == 2.0
  m = x(10000)
  assert abs(m.mean() - 0.5) < 0.1
  assert abs(m.std() - 2.) < 0.1

def test_lognormal():
  
  x = variate_generator(mt19937(), lognormal('float64', mean=0.5, sigma=2.0))
  assert x.distribution.mean == 0.5
  assert x.distribution.sigma == 2.0
  m = x(10000)
  assert abs(m.mean() - 0.5) < 0.1

