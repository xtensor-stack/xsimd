#ifndef XSIMD_SSE4_1_REGISTER_HPP
#define XSIMD_SSE4_1_REGISTER_HPP

#include "./xsimd_sse3_register.hpp"

namespace xsimd {

  struct sse4_1 : sse3 {};

  namespace types {

    XSIMD_DECLARE_SIMD_REGISTER_ALIAS(sse4_1, sse3);

  }
}
#endif
