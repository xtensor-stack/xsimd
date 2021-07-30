#ifndef XSIMD_FMA5_HPP
#define XSIMD_FMA5_HPP

#include "../types/xsimd_fma5_register.hpp"


namespace xsimd {

  namespace kernel {
    using namespace types;

    // fnma
    template<class A> batch<float, A> fnma(batch<float, A> const& x, batch<float, A> const& y, batch<float, A> const& z, requires_arch<fma5>) {
      return _mm256_fnmadd_ps(x, y, z);
    }

    template<class A> batch<double, A> fnma(batch<double, A> const& x, batch<double, A> const& y, batch<double, A> const& z, requires_arch<fma5>) {
      return _mm256_fnmadd_pd(x, y, z);
    }

    // fnms
    template<class A> batch<float, A> fnms(batch<float, A> const& x, batch<float, A> const& y, batch<float, A> const& z, requires_arch<fma5>) {
      return _mm256_fnmsub_ps(x, y, z);
    }

    template<class A> batch<double, A> fnms(batch<double, A> const& x, batch<double, A> const& y, batch<double, A> const& z, requires_arch<fma5>) {
      return _mm256_fnmsub_pd(x, y, z);
    }

    // fma
    template<class A> batch<float, A> fma(batch<float, A> const& x, batch<float, A> const& y, batch<float, A> const& z, requires_arch<fma5>) {
      return _mm256_fmadd_ps(x, y, z);
    }

    template<class A> batch<double, A> fma(batch<double, A> const& x, batch<double, A> const& y, batch<double, A> const& z, requires_arch<fma5>) {
      return _mm256_fmadd_pd(x, y, z);
    }

    // fms
    template<class A> batch<float, A> fms(batch<float, A> const& x, batch<float, A> const& y, batch<float, A> const& z, requires_arch<fma5>) {
      return _mm256_fmsub_ps(x, y, z);
    }

    template<class A> batch<double, A> fms(batch<double, A> const& x, batch<double, A> const& y, batch<double, A> const& z, requires_arch<fma5>) {
      return _mm256_fmsub_pd(x, y, z);
    }


  }

}

#endif
