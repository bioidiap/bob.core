/**
 * @date Wed Feb 9 12:26:11 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines easy-to-use Blitz++ cast functions for Bob
 * applications.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_CORE_CAST_H
#define BOB_CORE_CAST_H

#include <bob.core/assert.h>
#include <blitz/array.h>

namespace bob { namespace core {

  template<typename T, typename U> T cast(const U& in) {
    return static_cast<T>(in);
  }

  namespace array {

    template<typename T, typename U>
      blitz::Array<T,1> cast(const blitz::Array<U,1>& in) {
        bob::core::array::assertZeroBase(in);
        blitz::Array<T,1> out(in.extent(0));
        for( int i=0; i<in.extent(0); ++i)
          out(i) = bob::core::cast<T>( in(i));
        return out;
      }

    template<typename T, typename U>
      blitz::Array<T,2> cast(const blitz::Array<U,2>& in) {
        bob::core::array::assertZeroBase(in);
        blitz::Array<T,2> out(in.extent(0),in.extent(1));
        for( int i=0; i<in.extent(0); ++i)
          for( int j=0; j<in.extent(1); ++j)
            out(i,j) = bob::core::cast<T>( in(i,j) );
        return out;
      }

    template<typename T, typename U>
      blitz::Array<T,3> cast(const blitz::Array<U,3>& in) {
        bob::core::array::assertZeroBase(in);
        blitz::Array<T,3> out(in.extent(0),in.extent(1),in.extent(2));
        for( int i=0; i<in.extent(0); ++i)
          for( int j=0; j<in.extent(1); ++j)
            for( int k=0; k<in.extent(2); ++k)
              out(i,j,k) = bob::core::cast<T>( in(i,j,k) );
        return out;
      }

    template<typename T, typename U>
      blitz::Array<T,4> cast(const blitz::Array<U,4>& in) {
        bob::core::array::assertZeroBase(in);
        blitz::Array<T,4> out(in.extent(0),in.extent(1),in.extent(2),in.extent(3));
        for( int i=0; i<in.extent(0); ++i)
          for( int j=0; j<in.extent(1); ++j)
            for( int k=0; k<in.extent(2); ++k)
              for( int l=0; l<in.extent(3); ++l)
                out(i,j,k,l) = bob::core::cast<T>( in(i,j,k,l) );
        return out;
      }

  }

}}

#endif /* BOB_CORE_CAST_H */
