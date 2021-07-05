#ifndef XSIMD_FMA5_HPP
#define XSIMD_FMA5_HPP

#include "../types/xsimd_fma5_register.hpp"


namespace xsimd {

  namespace kernel {
    using namespace types;

    // fnma
    template<class A> batch<float, A> fnma(simd_register<float, A> const& x, simd_register<float, A> const& y, simd_register<float, A> const& z, requires<fma>) {
      return _mm256_fnmadd_ps(x, y, z);
    }

    template<class A> batch<double, A> fnma(simd_register<double, A> const& x, simd_register<double, A> const& y, simd_register<double, A> const& z, requires<fma>) {
      return _mm256_fnmadd_pd(x, y, z);
    }

    // fnms
    template<class A> batch<float, A> fnms(simd_register<float, A> const& x, simd_register<float, A> const& y, simd_register<float, A> const& z, requires<fma>) {
      return _mm256_fnmsub_ps(x, y, z);
    }

    template<class A> batch<double, A> fnms(simd_register<double, A> const& x, simd_register<double, A> const& y, simd_register<double, A> const& z, requires<fma>) {
      return _mm256_fnmsub_pd(x, y, z);
    }

    // fma
    template<class A> batch<float, A> fma(simd_register<float, A> const& x, simd_register<float, A> const& y, simd_register<float, A> const& z, requires<fma>) {
      return _mm256_fmadd_ps(x, y, z);
    }

    template<class A> batch<double, A> fma(simd_register<double, A> const& x, simd_register<double, A> const& y, simd_register<double, A> const& z, requires<fma>) {
      return _mm256_fmadd_pd(x, y, z);
    }

    // fms
    template<class A> batch<float, A> fms(simd_register<float, A> const& x, simd_register<float, A> const& y, simd_register<float, A> const& z, requires<fma>) {
      return _mm256_fmsub_ps(x, y, z);
    }

    template<class A> batch<double, A> fms(simd_register<double, A> const& x, simd_register<double, A> const& y, simd_register<double, A> const& z, requires<fma>) {
      return _mm256_fmsub_pd(x, y, z);
    }


  }

}

#endif
