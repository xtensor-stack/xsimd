#ifndef XSIMD_FMA3_HPP
#define XSIMD_FMA3_HPP

#include "../types/xsimd_sse_register.hpp"


namespace xsimd {

  namespace kernel {
    using namespace types;

    // fnma
    template<class A> batch<float, A> fnma(simd_register<float, A> const& x, simd_register<float, A> const& y, simd_register<float, A> const& z, requires_arch<fma4>) {
      return _mm_nmacc_ps(x, y, z);
    }

    template<class A> batch<double, A> fnma(simd_register<double, A> const& x, simd_register<double, A> const& y, simd_register<double, A> const& z, requires_arch<fma4>) {
      return _mm_nmacc_pd(x, y, z);
    }

    // fnms
    template<class A> batch<float, A> fnms(simd_register<float, A> const& x, simd_register<float, A> const& y, simd_register<float, A> const& z, requires_arch<fma4>) {
      return _mm_nmsub_ps(x, y, z);
    }

    template<class A> batch<double, A> fnms(simd_register<double, A> const& x, simd_register<double, A> const& y, simd_register<double, A> const& z, requires_arch<fma4>) {
      return _mm_nmsub_pd(x, y, z);
    }

    // fma
    template<class A> batch<float, A> fma(simd_register<float, A> const& x, simd_register<float, A> const& y, simd_register<float, A> const& z, requires_arch<fma4>) {
      return _mm_macc_ps(x, y, z);
    }

    template<class A> batch<double, A> fma(simd_register<double, A> const& x, simd_register<double, A> const& y, simd_register<double, A> const& z, requires_arch<fma4>) {
      return _mm_macc_pd(x, y, z);
    }

    // fms
    template<class A> batch<float, A> fms(simd_register<float, A> const& x, simd_register<float, A> const& y, simd_register<float, A> const& z, requires_arch<fma4>) {
      return _mm_msub_ps(x, y, z);
    }

    template<class A> batch<double, A> fms(simd_register<double, A> const& x, simd_register<double, A> const& y, simd_register<double, A> const& z, requires_arch<fma4>) {
      return _mm_msub_pd(x, y, z);
    }
  }

}

#endif

