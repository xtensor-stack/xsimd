#ifndef XSIMD_SSE4_2_REGISTER_HPP
#define XSIMD_SSE4_2_REGISTER_HPP

#include "./xsimd_sse4_1_register.hpp"

namespace xsimd {

  struct sse4_2 : sse4_1 {};

  namespace types {

    XSIMD_DECLARE_SIMD_REGISTER_ALIAS(sse4_2, sse4_1);

  }
}
#endif

