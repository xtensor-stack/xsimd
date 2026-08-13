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

#ifndef XSIMD_AVX512VNNI_AVX512VBMI2_HPP
#define XSIMD_AVX512VNNI_AVX512VBMI2_HPP

#include "../types/xsimd_avx512vnni_avx512vbmi2_register.hpp"
#include "./xsimd_avx512vnni_avx512bw.hpp"

namespace xsimd
{

    namespace kernel
    {

        using namespace types;

        // popcount
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 4>>
        XSIMD_INLINE batch<T, A> popcount(batch<T, A> const& self, requires_arch<avx512vnni<avx512vbmi2>>) noexcept
        {
            return popcount(self, avx512vnni<avx512bw> {});
        }
    }
}

#endif
