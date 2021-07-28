#ifndef XSIMD_SSE2_REGISTER_HPP
#define XSIMD_SSE2_REGISTER_HPP

#include "./xsimd_sse_register.hpp"

#if XSIMD_WITH_SSE2
#include <emmintrin.h>
#endif

namespace xsimd
{
    struct sse2 : sse
    {
        static constexpr bool supported() { return XSIMD_WITH_SSE2; }
        static constexpr bool available() { return true; }
        static constexpr unsigned version() { return generic::version(1, 2, 0); }
        static constexpr std::size_t alignment() { return 16; }
    };

#if XSIMD_WITH_SSE2
    namespace types
    {
        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(sse2, sse);
        XSIMD_DECLARE_SIMD_REGISTER(bool, sse2, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(signed char, sse2, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned char, sse2, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(char, sse2, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned short, sse2, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(short, sse2, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned int, sse2, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(int, sse2, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned long int, sse2, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(long int, sse2, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned long long int, sse2, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(long long int, sse2, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(double, sse2, __m128d);
    }
#endif
}

#endif

