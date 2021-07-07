#ifndef XSIMD_AVX512F_REGISTER_HPP
#define XSIMD_AVX512F_REGISTER_HPP

#include "./xsimd_generic_arch.hpp"

namespace xsimd {

  struct avx512f : generic {
    static constexpr bool supported() { return XSIMD_WITH_AVX512F; }
    static constexpr bool available() { return true; }
    static constexpr unsigned version() { return generic::version(3, 1, 0); }
    static constexpr std::size_t alignment() { return 64; }
  };

#if XSIMD_WITH_AVX512F
  namespace types {

    XSIMD_DECLARE_SIMD_REGISTER(bool, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(signed char, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(unsigned char, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(char, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(unsigned short, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(short, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(unsigned int, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(int, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(unsigned long int, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(long int, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(unsigned long long int, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(long long int, avx512f, __m512i);
    XSIMD_DECLARE_SIMD_REGISTER(float, avx512f, __m512);
    XSIMD_DECLARE_SIMD_REGISTER(double, avx512f, __m512d);

  }
#endif
}
#endif

