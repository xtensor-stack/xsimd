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

#ifndef XSIMD_FMA5_REGISTER_HPP
#define XSIMD_FMA5_REGISTER_HPP

#include "./xsimd_avx2_register.hpp"

namespace xsimd
{

    /**
     * @ingroup arch
     *
     * AVX2 + FMA instructions
     */
    struct fma5 : avx2
    {
        static constexpr bool supported() noexcept { return XSIMD_WITH_FMA5; }
        static constexpr bool available() noexcept { return true; }
        static constexpr unsigned version() noexcept { return generic::version(2, 3, 0); }
        static constexpr char const* name() noexcept { return "avx2+fma"; }
    };

#if XSIMD_WITH_FMA5
    namespace types
    {

        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(fma5, avx2);

    }
#endif
}
#endif
