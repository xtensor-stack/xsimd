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

#ifndef XSIMD_FMA3_REGISTER_HPP
#define XSIMD_FMA3_REGISTER_HPP

#include "./xsimd_sse4_2_register.hpp"

namespace xsimd
{

    /**
     * @ingroup arch
     *
     * SSE4.2 + FMA instructions
     */
    struct fma3 : sse4_2
    {
        static constexpr bool supported() noexcept { return XSIMD_WITH_FMA3; }
        static constexpr bool available() noexcept { return true; }
        static constexpr unsigned version() noexcept { return generic::version(1, 5, 0); }
        static constexpr char const* name() noexcept { return "sse4.2+fma"; }
    };

#if XSIMD_WITH_FMA3
    namespace types
    {

        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(fma3, sse4_2);

    }
#endif
}
#endif
