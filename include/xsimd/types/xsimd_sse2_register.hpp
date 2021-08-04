#ifndef XSIMD_SSE2_REGISTER_HPP
#define XSIMD_SSE2_REGISTER_HPP

#include "./xsimd_register.hpp"
#include "./xsimd_generic_arch.hpp"

#if XSIMD_WITH_SSE2
#include <xmmintrin.h>
#include <emmintrin.h>
#endif

namespace xsimd
{
    struct sse2 : generic
    {
        static constexpr bool supported() { return XSIMD_WITH_SSE2; }
        static constexpr bool available() { return true; }
        static constexpr bool requires_alignment() { return true; }
        static constexpr unsigned version() { return generic::version(1, 2, 0); }
        static constexpr std::size_t alignment() { return 16; }
        static constexpr char const* name() { return "sse2"; }
    };

#if XSIMD_WITH_SSE2
    namespace types
    {
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
        XSIMD_DECLARE_SIMD_REGISTER(float, sse2, __m128);
        XSIMD_DECLARE_SIMD_REGISTER(double, sse2, __m128d);
    }
#endif
}

#endif

