#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz']))
from bob.blitz.extension import Extension, Library, build_ext

import os
package_dir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.join(package_dir, 'bob', 'core')

packages = ['blitz >= 0.10', 'boost']
version = '2.0.0a0'

setup(

    name='bob.core',
    version=version,
    description='Core utilities required on all Bob modules',
    url='http://github.com/bioidiap/bob.core',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'bob.blitz',
    ],

    namespace_packages=[
      "bob",
      ],

    ext_modules = [
      Extension("bob.core.version",
        [
          "bob/core/version.cpp",
          ],
        version = version,
        packages = packages,
        ),
      Library("bob_core",
        [
          "bob/core/cpp/logging.cpp",
        ],
        package_directory = package_dir,
        target_directory = target_dir,
        version = version,
        packages = packages,
      ),
      Extension("bob.core._convert",
        [
          "bob/core/convert.cpp",
          ],
        version = version,
        packages = packages,
        ),
      Extension("bob.core._logging",
        [
          "bob/core/logging.cpp",
          ],
        version = version,
        packages = packages,
        libraries = ['bob_core'],
        boost_modules = ['iostreams', 'filesystem'],
        ),
      Extension("bob.core.random._library",
        [
          "bob/core/random/mt19937.cpp",
          "bob/core/random/uniform.cpp",
          "bob/core/random/normal.cpp",
          "bob/core/random/lognormal.cpp",
          "bob/core/random/gamma.cpp",
          "bob/core/random/binomial.cpp",
          "bob/core/random/discrete.cpp",
          "bob/core/random/main.cpp",
          ],
        version = version,
        packages = packages,
        ),
      ],

    cmdclass = {
      'build_ext': build_ext
    },

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
