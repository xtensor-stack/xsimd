#ifndef XSIMD_GENERIC_HPP
#define XSIMD_GENERIC_HPP

#include "../types/xsimd_generic_arch.hpp"


namespace xsimd {

  namespace kernel {
    using namespace types;

    // fma
    template<class A, class T> batch<T, A> fma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires<generic>) {
      return x * y + z;
    }

    // fms
    template<class A, class T> batch<T, A> fms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires<generic>) {
      return x * y - z;
    }

    // fnma
    template<class A, class T> batch<T, A> fnma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires<generic>) {
      return -x * y + z;
    }

    // fnms
    template<class A, class T> batch<T, A> fnms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires<generic>) {
      return -x * y - z;
    }
  }

}

#endif

