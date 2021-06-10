#ifndef XSIMD_SSSE3_HPP
#define XSIMD_SSSE3_HPP

#include "../types/xsimd_ssse3_register.hpp"

#include <tmmintrin.h>


namespace xsimd {

  namespace kernel {
    using namespace types;

    // abs
    template<class A, class T, typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, void>::type>
    batch<T, A> abs(batch<T, A> const& self, requires<ssse3>) {
      switch(sizeof(T)) {
        case 1: return _mm_abs_epi8(self);
        case 2: return _mm_abs_epi16(self);
        case 4: return _mm_abs_epi32(self);
        case 8: return _mm_abs_epi64(self);
        default: assert(false && "unsupported arch/op combination"); return {};
      }
    }

  }

}

#endif

