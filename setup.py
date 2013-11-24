#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['xbob.blitz']))
from xbob.blitz.extension import Extension

import os
package_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(package_dir, 'xbob', 'core', 'include')
include_dirs = [package_dir]

packages = ['bob-core >= 1.3']
version = '2.0.0a0'

setup(

    name='xbob.core',
    version=version,
    description='Bindings for bob.core',
    url='http://github.com/anjos/xbob.core',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'xbob.blitz',
    ],

    namespace_packages=[
      "xbob",
      ],

    ext_modules = [
      Extension("xbob.core._externals",
        [
          "xbob/core/externals.cpp",
          ],
        version = version,
        packages = packages,
        include_dirs = include_dirs,
        ),
      Extension("xbob.core._convert",
        [
          "xbob/core/convert.cpp",
          ],
        version = version,
        packages = packages,
        include_dirs = include_dirs,
        ),
      Extension("xbob.core._logging",
        [
          "xbob/core/logging.cpp",
          ],
        version = version,
        packages = packages,
        include_dirs = include_dirs,
        ),
      Extension("xbob.core.random._library",
        [
          "xbob/core/random/mt19937.cpp",
          "xbob/core/random/uniform.cpp",
          "xbob/core/random/normal.cpp",
          "xbob/core/random/lognormal.cpp",
          "xbob/core/random/gamma.cpp",
          "xbob/core/random/binomial.cpp",
          "xbob/core/random/main.cpp",
          ],
        version = version,
        packages = packages,
        include_dirs = include_dirs,
        ),
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
