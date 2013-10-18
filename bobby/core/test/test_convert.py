#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 18 Oct 13:50:08 2013 

"""Tests for core conversion functions
"""

from .. import convert
import numpy

def test_default_ranges(self):

  x = numpy.array(range(6), 'uint8').reshape(2,3)
  c = convert(x, 'uint16')
  assert numpy.array_equal(x.astype('uint16'), c)

def test_from_range(self):

  x = numpy.array(range(6), 'uint8').reshape(2,3)
  c = convert(x, 'uint16', source_range=(0,255))
  assert numpy.array_equal(x.astype('float64'), c)

def test_to_range(self):

  x = numpy.array(range(6), 'uint8').reshape(2,3)
  c = convert(x, 'float64', dest_range=(0.,255.))
  assert numpy.array_equal(x.astype('float64'), c)

def test_from_and_to_range(self):

  x = numpy.array(range(6), 'uint8').reshape(2,3)
  c = convert(x, 'float64', source_range=(0,255), dest_range=(0.,255.))
  assert numpy.array_equal(x.astype('float64'), c)
