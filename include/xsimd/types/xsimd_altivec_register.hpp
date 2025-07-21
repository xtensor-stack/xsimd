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

#ifndef XSIMD_ALTIVEC_REGISTER_HPP
#define XSIMD_ALTIVEC_REGISTER_HPP

#include "./xsimd_common_arch.hpp"
#include "./xsimd_register.hpp"

#if XSIMD_WITH_ALTIVEC
#include <altivec.h>
#endif

namespace xsimd
{
    /**
     * @ingroup architectures
     *
     * Altivec instructions
     */
    struct altivec : common
    {
        static constexpr bool supported() noexcept { return XSIMD_WITH_ALTIVEC; }
        static constexpr bool available() noexcept { return true; }
        static constexpr bool requires_alignment() noexcept { return true; }
        static constexpr std::size_t alignment() noexcept { return 16; }
        static constexpr char const* name() noexcept { return "altivec"; }
    };

#if XSIMD_WITH_ALTIVEC
    namespace types
    {

#define XSIMD_DECLARE_SIMD_BOOL_ALTIVEC_REGISTER(T, Tb)              \
    template <>                                                      \
    struct get_bool_simd_register<T, altivec>                        \
    {                                                                \
        struct type                                                  \
        {                                                            \
            using register_type = __vector __bool Tb;                \
            register_type data;                                      \
            type() = default;                                        \
            type(register_type r)                                    \
                : data(r)                                            \
            {                                                        \
            }                                                        \
            operator register_type() const noexcept { return data; } \
        };                                                           \
    };                                                               \
    XSIMD_DECLARE_SIMD_REGISTER(T, altivec, __vector T)

        XSIMD_DECLARE_SIMD_BOOL_ALTIVEC_REGISTER(signed char, char);
        XSIMD_DECLARE_SIMD_BOOL_ALTIVEC_REGISTER(unsigned char, char);
        XSIMD_DECLARE_SIMD_BOOL_ALTIVEC_REGISTER(char, char);
        XSIMD_DECLARE_SIMD_BOOL_ALTIVEC_REGISTER(unsigned short, short);
        XSIMD_DECLARE_SIMD_BOOL_ALTIVEC_REGISTER(short, short);
        XSIMD_DECLARE_SIMD_BOOL_ALTIVEC_REGISTER(unsigned int, int);
        XSIMD_DECLARE_SIMD_BOOL_ALTIVEC_REGISTER(int, int);
        XSIMD_DECLARE_SIMD_BOOL_ALTIVEC_REGISTER(unsigned long, long);
        XSIMD_DECLARE_SIMD_BOOL_ALTIVEC_REGISTER(long, long);
        XSIMD_DECLARE_SIMD_BOOL_ALTIVEC_REGISTER(float, int);

#undef XSIMD_DECLARE_SIMD_BOOL_ALTIVEC_REGISTER
    }
#endif
}

#endif
