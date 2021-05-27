#ifndef XSIMD_SSE2_REGISTER_HPP
#define XSIMD_SSE2_REGISTER_HPP

#include "./xsimd_sse_register.hpp"

#include <xmmintrin.h>

namespace xsimd {

  struct sse2 : sse {
    static constexpr std::size_t alignment = 16;
  };

  namespace types {

    XSIMD_DECLARE_SIMD_REGISTER_ALIAS(sse2, sse);

  }
}
#endif

