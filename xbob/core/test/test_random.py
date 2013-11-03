#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Sat  2 Nov 10:05:23 2013 

"""Tests for xbob.core.random
"""

from ..random import mt19937, uniform

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
