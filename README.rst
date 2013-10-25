.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Thu 29 Aug 2013 16:07:57 CEST

==============================
 Python bindings for bob.core
==============================

This package contains a set of Pythonic bindings for Bob's core packages and
functionality.

Installation
------------

Install it through normal means, via PyPI or use ``zc.buildout`` to bootstrap
the package and run test units.

Testing
-------

Run the test suite using ``./bin/nosetests``.

.. warning::

   If Bob <= 1.2.1 is installed on your python path, nose will automatically
   load the old version of the insulate plugin available in Bob, which will
   trigger the loading of incompatible shared libraries (from Bob itself), in
   to your working binary. This will cause a stack corruption. Either remove
   the centrally installed version of Bob, or build your own version of Python
   in which Bob <= 1.2.1 is not installed.
