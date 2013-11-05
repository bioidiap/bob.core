.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue 15 Oct 17:41:52 2013

.. testsetup:: coretest

   import numpy
   import xbob.core

============
 User Guide
============

Array Conversion
----------------

The function :py:func:`xbob.core.convert` allows you to convert objects of type
:py:class:`numpy.ndarray` between different types, with range compression or
decompression. For example, here we demonstrate a conversion using default
ranges. In this type of conversion, our implementation will assume that the
source array contains values within the range of ``uint8_t`` numbers and will
expand it to the range of ``uint16_t`` numbers, as desired by the programmer:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> x = numpy.array([0,255,0,255,0,255], 'uint8').reshape(2,3)
   >>> x
   array([[  0, 255,   0],
          [255,   0, 255]], dtype=uint8)
   >>> xbob.core.convert(x, 'uint16')
   array([[    0, 65535,     0],
          [65535,     0, 65535]], dtype=uint16)


The user can optionally specify source, destination ranges or both. For
example:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> x = numpy.array([0, 10, 20, 30, 40], 'uint8')
   >>> xbob.core.convert(x, 'float64', source_range=(0,40), dest_range=(0.,1.))
   array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

Any range not specified is assumed to default on the type range.


Random Number Generation
------------------------

You can build a new random number generator (RNG) of type
:py:class:`xbob.core.random.mt19937` using one of two possible ways:

1. Use the default constructor, which initializes with the default seed:

   .. doctest:: coretest
      :options: +NORMALIZE_WHITESPACE

      >>> xbob.core.random.mt19937()
      xbob.core.random.mt19937()

2. Pass a seed while initializing:

   .. doctest:: coretest
      :options: +NORMALIZE_WHITESPACE

      >>> rng = xbob.core.random.mt19937(34)

RNGs can be compared for equality. The ``==`` operator checks if both
generators are on the exact same state and would generate the same sequence of
numbers when exposed to the same distributions. For example:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> rng1 = xbob.core.random.mt19937(111)
   >>> rng2 = xbob.core.random.mt19937(111)
   >>> rng1 == rng2
   True
   >>> rng3 = xbob.core.random.mt19937(12)
   >>> rng1 == rng3
   False

The seed can be re-initialized at any point in time, which can be used to sync
two RNGs:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> rng3.seed(111)
   >>> rng1 == rng3
   True

Distributions skew numbers produced by the RNG so they look like the
parameterized distribution. By calling a distribution with an RNG, one
effectively generates random numbers:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> rng = xbob.core.random.mt19937()
   >>> # creates an uniform distribution of integers inside [0, 10]
   >>> u = xbob.core.random.uniform(int, 0, 10)
   >>> u(rng) # doctest: +SKIP
   8

At our reference guide (see below), you will find more implemented
distributions you can use on your programs. To simplify the task of generating
random numbers, we provide a class that mimics the behavior of
``boost::random::variate_generator``, in Python:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> ugen = xbob.core.random.variate_generator(rng, u)
   >>> ugen() # doctest: +SKIP
   6

You can also pass an optional shape when you call the variate generator, in
which case it generates a :py:class:`numpy.ndarray` of the specified size:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> ugen((3,3)) # doctest: +SKIP
   array([[ 3,  1,  6],
          [ 3,  2,  6],
          [10, 10, 10]])

Reference
---------

This section includes information for using the pure Python API of
``xbob.core``.

.. autofunction:: xbob.core.get_include

.. autofunction:: xbob.core.convert

.. autoclass:: xbob.core.random.mt19937

.. autoclass:: xbob.core.random.uniform

.. autoclass:: xbob.core.random.normal

.. autoclass:: xbob.core.random.lognormal

.. autoclass:: xbob.core.random.gamma

.. autoclass:: xbob.core.random.binomial

.. autoclass:: xbob.core.random.variate_generator
