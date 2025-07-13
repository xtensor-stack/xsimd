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
        XSIMD_DECLARE_SIMD_REGISTER(signed char, altivec, __vector signed char);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned char, altivec, __vector unsigned char);
        XSIMD_DECLARE_SIMD_REGISTER(char, altivec, __vector char);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned short, altivec, __vector unsigned short);
        XSIMD_DECLARE_SIMD_REGISTER(short, altivec, __vector short);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned int, altivec, __vector unsigned int);
        XSIMD_DECLARE_SIMD_REGISTER(int, altivec, __vector int);
        XSIMD_DECLARE_SIMD_REGISTER(float, altivec, __vector float);
    }
#endif
}

#endif
