/**
 * @date Wed Feb 9 12:26:11 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines functions which add std::complex support to the
 * static_cast function.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_CORE_COMPLEX_CAST_H
#define BOB_CORE_COMPLEX_CAST_H

#include <complex>
#include <stdint.h>
#include <bob.core/cast.h>

namespace bob { namespace core {

  // Complex to regular
# define COMPLEX_TO_REGULAR_DECL(COMP, REG) template<> \
  REG cast<REG, COMP>( const COMP& in);

# define COMPLEX_TO_REGULAR_FULL_DECL(COMP) \
  COMPLEX_TO_REGULAR_DECL(COMP, bool) \
  COMPLEX_TO_REGULAR_DECL(COMP, int8_t) \
  COMPLEX_TO_REGULAR_DECL(COMP, int16_t) \
  COMPLEX_TO_REGULAR_DECL(COMP, int32_t) \
  COMPLEX_TO_REGULAR_DECL(COMP, int64_t) \
  COMPLEX_TO_REGULAR_DECL(COMP, uint8_t) \
  COMPLEX_TO_REGULAR_DECL(COMP, uint16_t) \
  COMPLEX_TO_REGULAR_DECL(COMP, uint32_t) \
  COMPLEX_TO_REGULAR_DECL(COMP, uint64_t) \
  COMPLEX_TO_REGULAR_DECL(COMP, float) \
  COMPLEX_TO_REGULAR_DECL(COMP, double) \
  COMPLEX_TO_REGULAR_DECL(COMP, long double)

  COMPLEX_TO_REGULAR_FULL_DECL(std::complex<float>)
    COMPLEX_TO_REGULAR_FULL_DECL(std::complex<double>)
    COMPLEX_TO_REGULAR_FULL_DECL(std::complex<long double>)

    // Complex to complex
# define COMPLEX_TO_COMPLEX_DECL(FROM, TO) template<> \
    TO cast<TO, FROM>( const FROM& in);

# define COMPLEX_TO_COMPLEX_FULL_DECL(COMP) \
    COMPLEX_TO_COMPLEX_DECL(COMP, std::complex<float>) \
    COMPLEX_TO_COMPLEX_DECL(COMP, std::complex<double>) \
    COMPLEX_TO_COMPLEX_DECL(COMP, std::complex<long double>)

    COMPLEX_TO_COMPLEX_FULL_DECL(std::complex<float>)
    COMPLEX_TO_COMPLEX_FULL_DECL(std::complex<double>)
    COMPLEX_TO_COMPLEX_FULL_DECL(std::complex<long double>)

}}

#endif /* BOB_CORE_COMPLEX_CAST_H */
