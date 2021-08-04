#ifndef XSIMD_AVX512CD_REGISTER_HPP
#define XSIMD_AVX512CD_REGISTER_HPP

#include "./xsimd_avx512cd_register.hpp"

namespace xsimd {

  struct avx512cd : avx512f {
    static constexpr bool supported() { return XSIMD_WITH_AVX512BW; }
    static constexpr bool available() { return true; }
    static constexpr unsigned version() { return generic::version(3, 2, 0); }
    static constexpr char const* name() { return "avx512cd"; }
  };

#if XSIMD_WITH_AVX512BW

namespace types {
template <class T> struct get_bool_simd_register<T, avx512cd> {
  using type = simd_avx512_bool_register<T>;
};

    XSIMD_DECLARE_SIMD_REGISTER_ALIAS(avx512cd, avx512f);

  }
#endif
}
#endif

