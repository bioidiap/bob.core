/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 30 Oct 07:40:47 2013
 *
 * @brief C/C++-API for the random module
 */

#ifndef BOB_CORE_RANDOM_H
#define BOB_CORE_RANDOM_H

#include <boost/random.hpp>

namespace bob { namespace core { namespace random {

  // for completeness, we provide the interfaces for other distributions through bob::core::random

  template <class UniformRandomNumberGenerator=double, class RealType=double>
    using uniform_01 = boost::uniform_01<UniformRandomNumberGenerator,RealType>;

  template <class IntType=int>
    using uniform_smallint_distribution = boost::uniform_smallint<IntType>;

  template <class IntType=int>
    using uniform_int_distribution = boost::uniform_int<IntType>;

  template <class RealType=double>
    using uniform_real_distribution = boost::uniform_real<RealType>;

  template <class RealType=double>
    using gamma_distribution = boost::gamma_distribution<RealType>;

} } }

#include <boost/version.hpp>

#if BOOST_VERSION >= 105600

// Use the default implementations of BOOST
namespace bob { namespace core { namespace random {

  template <class RealType=double>
    using normal_distribution = boost::random::normal_distribution<RealType>;

  template <class RealType=double>
    using lognormal_distribution = boost::random::lognormal_distribution<RealType>;

  template<class IntType=int, class RealType=double>
    using binomial_distribution = boost::random::binomial_distribution<IntType, RealType>;

  template<class IntType, class WeightType>
    using discrete_distribution = boost::random::discrete_distribution<IntType, WeightType>;

} } } // namespaces

#else

// Use the copied implementations of boost 1.56
// where the bugs have been fixed
#include <bob.core/boost/normal_distribution.hpp>
#include <bob.core/boost/lognormal_distribution.hpp>
#include <bob.core/boost/binomial_distribution.hpp>
#include <bob.core/boost/discrete_distribution.hpp>

#endif // BOOST VERSION

#endif /* BOB_CORE_RANDOM_H */
