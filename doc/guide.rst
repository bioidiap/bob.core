.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue 15 Oct 17:41:52 2013

.. testsetup:: coretest

   import numpy
   import bob.core

============
 User Guide
============

Array Conversion
----------------


The function :py:func:`bob.core.convert` allows you to convert objects of type
:py:class:`numpy.ndarray` or :py:class:`bob.blitz.array` between different types, with range compression or
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
   >>> bob.core.convert(x, 'uint16')
   array([[    0, 65535,     0],
          [65535,     0, 65535]], dtype=uint16)


The user can optionally specify source, destination ranges or both. For
example:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> x = numpy.array([0, 10, 20, 30, 40], 'uint8')
   >>> bob.core.convert(x, 'float64', source_range=(0,40), dest_range=(0.,1.))
   array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

Any range not specified is assumed to default on the type range.


Random Number Generation
------------------------

You can build a new random number generator (RNG) of type
:py:class:`bob.core.random.mt19937` using one of two possible ways:

1. Use the default constructor, which initializes with the default seed:

   .. doctest:: coretest
      :options: +NORMALIZE_WHITESPACE

      >>> bob.core.random.mt19937()
      bob.core.random.mt19937()

2. Pass a seed while initializing:

   .. doctest:: coretest
      :options: +NORMALIZE_WHITESPACE

      >>> rng = bob.core.random.mt19937(34)

RNGs can be compared for equality. The ``==`` operator checks if both
generators are on the exact same state and would generate the same sequence of
numbers when exposed to the same distributions. For example:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> rng1 = bob.core.random.mt19937(111)
   >>> rng2 = bob.core.random.mt19937(111)
   >>> rng1 == rng2
   True
   >>> rng3 = bob.core.random.mt19937(12)
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
parametrized distribution. By calling a distribution with an RNG, one
effectively generates random numbers:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> rng = bob.core.random.mt19937()
   >>> # creates an uniform distribution of integers inside [0, 10]
   >>> u = bob.core.random.uniform(int, 0, 10)
   >>> u(rng) # doctest: +SKIP
   8

At our reference guide (see below), you will find more implemented
distributions you can use on your programs. To simplify the task of generating
random numbers, we provide a class that mimics the behavior of
``boost::random::variate_generator``, in Python:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> ugen = bob.core.random.variate_generator(rng, u)
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

Logging
-------

Bob provides logging capabilities to integrate log output from C++ using the python :py:mod:`logging` module.
In the :py:mod:`bob.core.log` module, there exist several functions to ease up the integration and the set-up of the logging module.

In an external python module you can use the :py:func:`bob.core.log.setup` function to generate and initialize a logger for you:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> logger = bob.core.log.setup("my.module.name")

This will instantiate a :py:class:`logging.Logger` object that you can use for logging information, such as:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> logger.info("This might be an interesting information...")

Now, when writing a python script, you can provide the command line option for your script, to increase the verbosity level of your script:

.. doctest:: coretest
   :options: +NORMALIZE_WHITESPACE

   >>> import argparse
   >>> parser = argparse.ArgumentParser()
   >>> # initialize command line arguments
   >>> # ...
   >>> bob.core.log.add_command_line_option(parser)
   >>> args = parser.parse_args([])
   >>> bob.core.log.set_verbosity_level(logger, args.verbose)

Of course, you can use several loggers and set different log levels for all loggers.
Anyways, the root logger ``logging.getLogger('bob')`` will always be affected by the last call.
