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

#ifndef XSIMD_LSX_REGISTER_HPP
#define XSIMD_LSX_REGISTER_HPP

#include "../config/xsimd_config.hpp"
#include "../utils/xsimd_type_traits.hpp"
#include "./xsimd_common_arch.hpp"
#include "./xsimd_register.hpp"

#include <cstddef>

#if XSIMD_WITH_LSX
#include <lsxintrin.h>
#endif

namespace xsimd
{
    /**
     * @ingroup architectures
     *
     * Loongson SIMD Extension (LSX).
     */
    struct lsx : common
    {
        static constexpr bool supported() noexcept { return XSIMD_WITH_LSX; }
        static constexpr bool available() noexcept { return true; }
        static constexpr bool requires_alignment() noexcept { return true; }
        static constexpr std::size_t alignment() noexcept { return 16; }
        static constexpr char const* name() noexcept { return "loongarch64+lsx"; }
    };

#if XSIMD_WITH_LSX
    namespace types
    {
        XSIMD_DECLARE_SIMD_REGISTER(signed char, lsx, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned char, lsx, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(char, lsx, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(short, lsx, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned short, lsx, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(int, lsx, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned int, lsx, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(long, lsx, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned long, lsx, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(long long, lsx, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned long long, lsx, __m128i);
        XSIMD_DECLARE_SIMD_REGISTER(float, lsx, __m128);
        XSIMD_DECLARE_SIMD_REGISTER(double, lsx, __m128d);

        template <class T>
        struct get_bool_simd_register<T, lsx>
        {
            using type = simd_register<xsimd::sized_uint_t<sizeof(T)>, lsx>;
        };
    }
#endif
}

#endif
