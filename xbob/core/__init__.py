from ._convert import convert
from . import log
from . import random
from . import version
from .version import module as __version__
from .version import api as __api_version__

def get_include():
  """Returns the directory containing the C/C++ API include directives"""

  return __import__('pkg_resources').resource_filename(__name__, 'include')

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
