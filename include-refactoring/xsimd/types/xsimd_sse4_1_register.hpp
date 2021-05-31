#ifndef XSIMD_SSE4_1_REGISTER_HPP
#define XSIMD_SSE4_1_REGISTER_HPP

#include "./xsimd_sse3_register.hpp"

namespace xsimd {

  struct sse4_1 : sse3 {
    static constexpr bool supported() { return XSIMD_WITH_SSE4_1; }
    static constexpr bool available() { return true; }
    static constexpr unsigned version() { return generic::version(1, 4, 1); }
  };

#if XSIMD_WITH_SSE4_1
  namespace types {

    XSIMD_DECLARE_SIMD_REGISTER_ALIAS(sse4_1, sse3);

  }
#endif
}
#endif
