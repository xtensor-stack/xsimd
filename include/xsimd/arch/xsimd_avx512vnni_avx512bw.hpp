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

#ifndef XSIMD_AVX512VNNI_AVX512_BW_HPP
#define XSIMD_AVX512VNNI_AVX512_BW_HPP

#include "../types/xsimd_avx512vnni_avx512bw_register.hpp"
#include "./xsimd_avx512bw.hpp"

namespace xsimd
{

    namespace kernel
    {

        using namespace types;

        // popcount
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 4>>
        XSIMD_INLINE batch<T, A> popcount(batch<T, A> const& self, requires_arch<avx512vnni<avx512bw>>) noexcept
        {
            // VPDPBUSD does in one uop what the VPMADDUBSW + VPMADDWD pair of
            // the avx512bw kernel does in two; the zero accumulator it needs
            // costs nothing, since the register copy is eliminated at rename
            return _mm512_dpbusd_epi32(_mm512_setzero_si512(), detail::popcount_bytes(self), _mm512_set1_epi8(1));
        }
    }
}

#endif
