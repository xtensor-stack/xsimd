#ifndef XSIMD_FMA5_REGISTER_HPP
#define XSIMD_FMA5_REGISTER_HPP

#include "./xsimd_avx2_register.hpp"

// Fake instruction set, maps to AVX2 + FMA
namespace xsimd {

  struct fma5 : avx2 {
    static constexpr bool supported() { return XSIMD_WITH_FMA5; }
    static constexpr bool available() { return true; }
    static constexpr unsigned version() { return generic::version(2, 3, 0); }
    static constexpr char const* name() { return "avx2+fma"; }
  };

#if XSIMD_WITH_FMA5
  namespace types {

    XSIMD_DECLARE_SIMD_REGISTER_ALIAS(fma5, avx2);

  }
#endif
}
#endif


