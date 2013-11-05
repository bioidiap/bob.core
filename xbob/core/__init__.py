from ._convert import convert
from . import log
from . import random

def get_include():
  """Returns the directory containing the C/C++ API include directives"""

  return __import__('pkg_resources').resource_filename(__name__, 'include')

__all__ = ['convert', 'log', 'random']
