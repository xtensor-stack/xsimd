/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_LASX_REGISTER_HPP
#define XSIMD_LASX_REGISTER_HPP

#include "./xsimd_lsx_register.hpp"

#if XSIMD_WITH_LASX && !XSIMD_WITH_LSX
#error "architecture inconsistency: lasx requires lsx"
#endif

#if XSIMD_WITH_LASX
#include <lasxintrin.h>
#endif

namespace xsimd
{
    /**
     * @ingroup architectures
     *
     * Loongson Advanced SIMD Extension (LASX).
     */
    struct lasx : common
    {
        static constexpr bool supported() noexcept { return XSIMD_WITH_LASX; }
        static constexpr bool available() noexcept { return true; }
        static constexpr bool requires_alignment() noexcept { return true; }
        static constexpr std::size_t alignment() noexcept { return 32; }
        static constexpr char const* name() noexcept { return "loongarch64+lasx"; }
    };

#if XSIMD_WITH_LASX
    namespace types
    {
        XSIMD_DECLARE_SIMD_REGISTER(signed char, lasx, __m256i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned char, lasx, __m256i);
        XSIMD_DECLARE_SIMD_REGISTER(char, lasx, __m256i);
        XSIMD_DECLARE_SIMD_REGISTER(short, lasx, __m256i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned short, lasx, __m256i);
        XSIMD_DECLARE_SIMD_REGISTER(int, lasx, __m256i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned int, lasx, __m256i);
        XSIMD_DECLARE_SIMD_REGISTER(long, lasx, __m256i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned long, lasx, __m256i);
        XSIMD_DECLARE_SIMD_REGISTER(long long, lasx, __m256i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned long long, lasx, __m256i);
        XSIMD_DECLARE_SIMD_REGISTER(float, lasx, __m256);
        XSIMD_DECLARE_SIMD_REGISTER(double, lasx, __m256d);

        template <class T>
        struct get_bool_simd_register<T, lasx>
        {
            using type = simd_register<xsimd::sized_uint_t<sizeof(T)>, lasx>;
        };
    }
#endif
}

#endif
