from ._convert import convert, __version__
from . import log
from . import random
from .random import __api_version__

def get_include():
  """Returns the directory containing the C/C++ API include directives"""

  return __import__('pkg_resources').resource_filename(__name__, 'include')
