#ifndef XSIMD_AVX512BW_REGISTER_HPP
#define XSIMD_AVX512BW_REGISTER_HPP

#include "./xsimd_avx512dq_register.hpp"

namespace xsimd {

  struct avx512bw : avx512dq {
    static constexpr bool supported() { return XSIMD_WITH_AVX512BW; }
    static constexpr bool available() { return true; }
    static constexpr unsigned version() { return generic::version(3, 4, 0); }
    static constexpr char const* name() { return "avx512bw"; }
  };

#if XSIMD_WITH_AVX512BW

namespace types {
template <class T> struct get_bool_simd_register<T, avx512bw> {
  using type = simd_avx512_bool_register<T>;
};

    XSIMD_DECLARE_SIMD_REGISTER_ALIAS(avx512bw, avx512dq);

  }
#endif
}
#endif

