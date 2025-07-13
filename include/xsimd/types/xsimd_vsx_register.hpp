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

#ifndef XSIMD_VSX_REGISTER_HPP
#define XSIMD_VSX_REGISTER_HPP

#include "./xsimd_common_arch.hpp"
#include "./xsimd_register.hpp"

#if XSIMD_WITH_VSX
#include <altivec.h>
#endif

namespace xsimd
{
    /**
     * @ingroup architectures
     *
     * VSX instructions
     */
    struct vsx : common
    {
        static constexpr bool supported() noexcept { return XSIMD_WITH_VSX; }
        static constexpr bool available() noexcept { return true; }
        static constexpr bool requires_alignment() noexcept { return true; }
        static constexpr std::size_t alignment() noexcept { return 16; }
        static constexpr char const* name() noexcept { return "vmx+vsx"; }
    };

#if XSIMD_WITH_VSX
    namespace types
    {

#define XSIMD_DECLARE_SIMD_BOOL_VSX_REGISTER(T, Tb)                  \
    template <>                                                      \
    struct get_bool_simd_register<T, vsx>                            \
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
    XSIMD_DECLARE_SIMD_REGISTER(T, vsx, __vector T)

        XSIMD_DECLARE_SIMD_BOOL_VSX_REGISTER(signed char, char);
        XSIMD_DECLARE_SIMD_BOOL_VSX_REGISTER(unsigned char, char);
        XSIMD_DECLARE_SIMD_BOOL_VSX_REGISTER(char, char);
        XSIMD_DECLARE_SIMD_BOOL_VSX_REGISTER(unsigned short, short);
        XSIMD_DECLARE_SIMD_BOOL_VSX_REGISTER(short, short);
        XSIMD_DECLARE_SIMD_BOOL_VSX_REGISTER(unsigned int, int);
        XSIMD_DECLARE_SIMD_BOOL_VSX_REGISTER(int, int);
        XSIMD_DECLARE_SIMD_BOOL_VSX_REGISTER(unsigned long, long);
        XSIMD_DECLARE_SIMD_BOOL_VSX_REGISTER(long, long);
        XSIMD_DECLARE_SIMD_BOOL_VSX_REGISTER(float, int);
        XSIMD_DECLARE_SIMD_BOOL_VSX_REGISTER(double, long);

#undef XSIMD_DECLARE_SIMD_BOOL_VSX_REGISTER
    }
#endif
}

#endif
