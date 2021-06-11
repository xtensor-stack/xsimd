#ifndef XSIMD_AVX_REGISTER_HPP
#define XSIMD_AVX_REGISTER_HPP

#include "./xsimd_generic_arch.hpp"

namespace xsimd {

  struct avx : generic {
    static constexpr bool supported() { return XSIMD_WITH_AVX; }
    static constexpr bool available() { return true; }
    static constexpr unsigned version() { return generic::version(2, 1, 0); }
    static constexpr std::size_t alignment() { return 32; }
  };
}

#if XSIMD_WITH_AVX

#include <immintrin.h>

namespace xsimd {
  namespace types {

    XSIMD_DECLARE_SIMD_REGISTER(float, avx, __m256);
    XSIMD_DECLARE_SIMD_REGISTER(double, avx, __m256d);

  }
}
#endif
#endif


