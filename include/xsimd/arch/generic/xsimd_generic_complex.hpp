#ifndef XSIMD_GENERIC_COMPLEX_HPP
#define XSIMD_GENERIC_COMPLEX_HPP

#include "./xsimd_generic_details.hpp"


namespace xsimd {

  namespace kernel {

    using namespace types;

    // arg
    template<class A, class T>
    batch<T, A> arg(batch<std::complex<T>, A> const& self, requires_arch<generic>) {
      return atan2(self.imag(), self.real());
    }


    // conj
    template<class A, class T> batch<std::complex<T>, A> conj(batch<std::complex<T>, A> const& self, requires_arch<generic>) {
      return {self.real(), - self.imag()};
    }

    // norm
    template<class A, class T> batch<T, A> norm(batch<std::complex<T>, A> const& self, requires_arch<generic>) {
      return {fma(self.real(), self.real(), self.imag() * self.imag())};
    }


    // proj
    template<class A, class T> batch<std::complex<T>, A> proj(batch<std::complex<T>, A> const& self, requires_arch<generic>) {
      using batch_type = batch<std::complex<T>, A>;
      using real_batch = typename batch_type::real_batch;
      auto cond = xsimd::isinf(self.real()) || xsimd::isinf(self.imag());
      return select(cond, batch_type(constants::infinity<real_batch>(), copysign(real_batch(0.), self.imag())), self);
    }


  }

}

#endif

