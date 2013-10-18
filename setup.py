#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
from distutils.extension import Extension

dist.Distribution(dict(setup_requires=['pypkg', 'numpy', 'blitz.array']))
import pypkg
import numpy
import blitz

# Pkg-config dependencies
blitz_pkg = pypkg.pkgconfig('blitz')
bob_pkg = pypkg.pkgconfig('bob-core')

# Add system include directories
extra_compile_args = []
system_includes = \
    [blitz.get_include()] + \
    blitz_pkg.include_directories() + \
    [numpy.get_include()]
for k in system_includes: extra_compile_args += ['-isystem', k]

# NumPy API macros necessary?
define_macros=[
    ("PY_ARRAY_UNIQUE_SYMBOL", blitz.get_numpy_api()),
    ("NO_IMPORT_ARRAY", "1"),
    ]

import numpy
from distutils.version import StrictVersion
if StrictVersion(numpy.__version__) >= StrictVersion('1.7'):
  define_macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))

# Compilation options
import platform
extra_compile_args += ['-O0', '-g']
if platform.system() == 'Darwin':
  extra_compile_args += ['-std=c++11', '-stdlib=libc++', '-Wno-#warnings']
else:
  extra_compile_args += ['-std=c++11']

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    name='bobby.core',
    version='2.0.0a0',
    description='Bindings for bob.core',
    url='http://github.com/anjos/bob.core',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'blitz.array',
    ],

    namespace_packages=[
      "bobby",
      ],

    ext_modules = [
      Extension("bobby.core._convert",
        [
          "bobby/core/convert.cpp",
          ],
        define_macros=define_macros,
        include_dirs=bob_pkg.include_directories(), 
        extra_compile_args=extra_compile_args,
        library_dirs=bob_pkg.library_directories(),
        libraries=bob_pkg.libraries(),
        language="c++",
        )
      ],

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],
 
    )
