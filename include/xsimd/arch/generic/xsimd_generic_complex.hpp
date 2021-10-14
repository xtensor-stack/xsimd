/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
* Copyright (c) Serge Guelton                                              *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_GENERIC_COMPLEX_HPP
#define XSIMD_GENERIC_COMPLEX_HPP

#include <complex>

#include "./xsimd_generic_details.hpp"

namespace xsimd {

  namespace kernel {

    using namespace types;

    // real
    template <class A, class T>
    batch<T, A> real(batch<T, A> const& self, requires_arch<generic>) {
      return self;
    }

    template <class A, class T>
    batch<T, A> real(batch<std::complex<T>, A> const& self, requires_arch<generic>) {
      return self.real();
    }

    // imag
    template <class A, class T>
    batch<T, A> imag(batch<T, A> const& /*self*/, requires_arch<generic>) {
      return batch<T, A>(T(0));
    }

    template <class A, class T>
    batch<T, A> imag(batch<std::complex<T>, A> const& self, requires_arch<generic>) {
      return self.imag();
    }
    
    // arg
    template<class A, class T>
    real_batch_type_t<batch<T, A>> arg(batch<T, A> const& self, requires_arch<generic>) {
      return atan2(imag(self), real(self));
    }

    // conj
    template<class A, class T>
    complex_batch_type_t<batch<T, A>> conj(batch<T, A> const& self, requires_arch<generic>) {
      return {real(self), - imag(self)};
    }

    // norm
    template<class A, class T>
    real_batch_type_t<batch<T, A>> norm(batch<T, A> const& self, requires_arch<generic>) {
      return {fma(real(self), real(self), imag(self) * imag(self))};
    }

    // proj
    template<class A, class T>
    complex_batch_type_t<batch<T, A>> proj(batch<T, A> const& self, requires_arch<generic>) {
      using batch_type = complex_batch_type_t<batch<T, A>>;
      using real_batch = typename batch_type::real_batch;
      using real_value_type = typename real_batch::value_type;
      auto cond = xsimd::isinf(real(self)) || xsimd::isinf(imag(self));
      return select(cond,
                    batch_type(constants::infinity<real_batch>(),
                               copysign(real_batch(real_value_type(0)), imag(self))),
                    batch_type(self));
    }

    template <class A, class T>
    batch_bool<T, A> isnan(batch<std::complex<T>, A> const& self, requires_arch<generic>) {
      return batch_bool<T, A>(isnan(self.real()) || isnan(self.imag()));
    }
  }
}

#endif

