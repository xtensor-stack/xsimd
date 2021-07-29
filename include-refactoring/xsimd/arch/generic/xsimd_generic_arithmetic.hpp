#ifndef XSIMD_GENERIC_ARITHMETIC_HPP
#define XSIMD_GENERIC_ARITHMETIC_HPP

#include "./xsimd_generic_details.hpp"


namespace xsimd {

  namespace kernel {

    using namespace types;

    // bitwise_lshift
    template<class A, class T, class/*=typename std::enable_if<std::is_integral<T>::value, void>::type*/>
    batch<T, A> bitwise_lshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) {
      return detail::apply([](T x, T y) { return x << y;}, self, other);
    }

    // bitwise_rshift
    template<class A, class T, class/*=typename std::enable_if<std::is_integral<T>::value, void>::type*/>
    batch<T, A> bitwise_rshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) {
      return detail::apply([](T x, T y) { return x >> y;}, self, other);
    }

    // div
    template<class A, class T, class=typename std::enable_if<std::is_integral<T>::value, void>::type>
    batch<T, A> div(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) {
      return detail::apply([](T x, T y) -> T { return x / y;}, self, other);
    }

    // fma
    template<class A, class T> batch<T, A> fma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires_arch<generic>) {
      return x * y + z;
    }

    template<class A, class T> batch<std::complex<T>, A> fma(batch<std::complex<T>, A> const& x, batch<std::complex<T>, A> const& y, batch<std::complex<T>, A> const& z, requires_arch<generic>) {
      auto res_r = fms(x.real(), y.real(), fms(x.imag(), y.imag(), z.real()));
      auto res_i = fma(x.real(), y.imag(), fma(x.imag(), y.real(), z.imag()));
      return {res_r, res_i};
    }

    // fms
    template<class A, class T> batch<T, A> fms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires_arch<generic>) {
      return x * y - z;
    }

    template<class A, class T> batch<std::complex<T>, A> fms(batch<std::complex<T>, A> const& x, batch<std::complex<T>, A> const& y, batch<std::complex<T>, A> const& z, requires_arch<generic>) {
      auto res_r = fms(x.real(), y.real(), fma(x.imag(), y.imag(), z.real()));
      auto res_i = fma(x.real(), y.imag(), fms(x.imag(), y.real(), z.imag()));
      return {res_r, res_i};
    }

    // fnma
    template<class A, class T> batch<T, A> fnma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires_arch<generic>) {
      return -x * y + z;
    }

    template<class A, class T> batch<std::complex<T>, A> fnma(batch<std::complex<T>, A> const& x, batch<std::complex<T>, A> const& y, batch<std::complex<T>, A> const& z, requires_arch<generic>) {
      auto res_r = - fms(x.real(), y.real(), fma(x.imag(), y.imag(), z.real()));
      auto res_i = - fma(x.real(), y.imag(), fms(x.imag(), y.real(), z.imag()));
      return {res_r, res_i};
    }

    // fnms
    template<class A, class T> batch<T, A> fnms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires_arch<generic>) {
      return -x * y - z;
    }

    template<class A, class T> batch<std::complex<T>, A> fnms(batch<std::complex<T>, A> const& x, batch<std::complex<T>, A> const& y, batch<std::complex<T>, A> const& z, requires_arch<generic>) {
      auto res_r = - fms(x.real(), y.real(), fms(x.imag(), y.imag(), z.real()));
      auto res_i = - fma(x.real(), y.imag(), fma(x.imag(), y.real(), z.imag()));
      return {res_r, res_i};
    }



    // mul
    template<class A, class T, class/*=typename std::enable_if<std::is_integral<T>::value, void>::type*/>
    batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) {
      return detail::apply([](T x, T y) -> T { return x * y;}, self, other);
    }

  }

}

#endif

