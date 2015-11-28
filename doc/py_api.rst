.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue 15 Oct 17:41:52 2013

.. testsetup:: coretest

   import numpy
   import bob.core

============
 Python API
============

This section includes information for using the pure Python API of ``bob.core``.

Logging
-------

.. autosummary::
   bob.core.log.setup
   bob.core.log.add_command_line_option
   bob.core.log.set_verbosity_level
   bob.core.log.reset


Random Numbers
--------------

.. autosummary::
   bob.core.random.mt19937
   bob.core.random.uniform
   bob.core.random.normal
   bob.core.random.lognormal
   bob.core.random.gamma
   bob.core.random.binomial
   bob.core.random.discrete
   bob.core.random.variate_generator


Functions
---------

.. autosummary::
   bob.core.convert
   bob.core.get_config


Details
-------

.. automodule:: bob.core.log
.. automodule:: bob.core.random
.. automodule:: bob.core
