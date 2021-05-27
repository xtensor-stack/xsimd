#ifndef XSIMD_SSE3_REGISTER_HPP
#define XSIMD_SSE3_REGISTER_HPP

#include "./xsimd_sse2_register.hpp"

#include <xmmintrin.h>

namespace xsimd {

  struct sse3 : sse2 {};

  namespace types {

    XSIMD_DECLARE_SIMD_REGISTER_ALIAS(sse3, sse2);

  }
}
#endif

