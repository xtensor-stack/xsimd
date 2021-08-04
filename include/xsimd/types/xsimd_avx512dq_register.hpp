#ifndef XSIMD_AVX512DQ_REGISTER_HPP
#define XSIMD_AVX512DQ_REGISTER_HPP

#include "./xsimd_avx512cd_register.hpp"

namespace xsimd {

  struct avx512dq : avx512cd {
    static constexpr bool supported() { return XSIMD_WITH_AVX512DQ; }
    static constexpr bool available() { return true; }
    static constexpr unsigned version() { return generic::version(3, 3, 0); }
    static constexpr char const* name() { return "avx512dq"; }
  };

#if XSIMD_WITH_AVX512DQ

namespace types {
template <class T> struct get_bool_simd_register<T, avx512dq> {
  using type = simd_avx512_bool_register<T>;
};

    XSIMD_DECLARE_SIMD_REGISTER_ALIAS(avx512dq, avx512cd);

  }
#endif
}
#endif

