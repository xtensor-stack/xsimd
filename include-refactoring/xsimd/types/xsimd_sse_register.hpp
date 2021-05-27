#ifndef XSIMD_SSE_REGISTER_HPP
#define XSIMD_SSE_REGISTER_HPP

#include "./xsimd_register.hpp"
#include "./xsimd_generic_arch.hpp"

#include <xmmintrin.h>

namespace xsimd {

  struct sse : generic {
    static constexpr std::size_t alignment = 8;
  };

  namespace types {

    XSIMD_DECLARE_SIMD_REGISTER(float, sse, __m128);
    XSIMD_DECLARE_SIMD_REGISTER(double, sse, __m128d);

  }
}
#endif

