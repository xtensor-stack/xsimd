#ifndef XSIMD_FMA3_REGISTER_HPP
#define XSIMD_FMA3_REGISTER_HPP

#include "./xsimd_sse4_2_register.hpp"

// Fake instruction set, maps to SSE4.2 + FMA
namespace xsimd {

  struct fma3 : sse4_2 {
    static constexpr bool supported() { return XSIMD_WITH_FMA3; }
    static constexpr bool available() { return true; }
    static constexpr unsigned version() { return generic::version(1, 5, 0); }
    static constexpr char const* name() { return "sse4.2+fma"; }
  };

#if XSIMD_WITH_FMA3
  namespace types {

    XSIMD_DECLARE_SIMD_REGISTER_ALIAS(fma3, sse4_2);

  }
#endif
}
#endif

