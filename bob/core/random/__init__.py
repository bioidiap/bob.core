from ._library import __doc__, mt19937, uniform, normal, lognormal, gamma, binomial, discrete
from ..version import api as __api_version__
import numpy

class variate_generator:
  """A pure-python version of the boost::variate_generator<> class

  Keyword parameters:

  engine
    An instance of the RNG you would like to use. This has to be an
    object of the class :py:class:`bob.core.random.mt19937`, already
    initialized.

  distribution
    The distribution to respect when generating scalars using the engine.
    The distribution object should be previously initialized.
  """

  def __init__(self, engine, distribution):

    self.engine = engine
    self.distribution = distribution

  def seed(self, value):
    """Resets the seed of the ``variate_generator`` with a (integer) value"""

    self.engine.seed(value)
    self.distribution.reset()

  def __call__(self, shape=None):
    """Use the ``()`` operator to generate a random scalar"""

    if shape is None:
      return self.distribution(self.engine)
    else:
      l = [self.distribution(self.engine) for k in range(numpy.prod(shape))]
      return numpy.array(l).reshape(shape)
