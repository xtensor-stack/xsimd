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

#ifndef XSIMD_AVX512VBMI2_HPP
#define XSIMD_AVX512VBMI2_HPP

#include <array>
#include <type_traits>

#include "../types/xsimd_avx512vbmi2_register.hpp"

namespace xsimd
{

    namespace kernel
    {
        using namespace types;

        // compress
        template <class A>
        XSIMD_INLINE batch<int16_t, A> compress(batch<int16_t, A> const& self, batch_bool<int16_t, A> const& mask, requires_arch<avx512vbmi2>) noexcept
        {
            return _mm512_maskz_compress_epi16(mask.mask(), self);
        }
        template <class A>
        XSIMD_INLINE batch<uint16_t, A> compress(batch<uint16_t, A> const& self, batch_bool<uint16_t, A> const& mask, requires_arch<avx512vbmi2>) noexcept
        {
            return _mm512_maskz_compress_epi16(mask.mask(), self);
        }
        template <class A>
        XSIMD_INLINE batch<int8_t, A> compress(batch<int8_t, A> const& self, batch_bool<int8_t, A> const& mask, requires_arch<avx512vbmi2>) noexcept
        {
            return _mm512_maskz_compress_epi8(mask.mask(), self);
        }
        template <class A>
        XSIMD_INLINE batch<uint8_t, A> compress(batch<uint8_t, A> const& self, batch_bool<uint8_t, A> const& mask, requires_arch<avx512vbmi2>) noexcept
        {
            return _mm512_maskz_compress_epi8(mask.mask(), self);
        }

        // expand
        template <class A>
        XSIMD_INLINE batch<int16_t, A> expand(batch<int16_t, A> const& self, batch_bool<int16_t, A> const& mask, requires_arch<avx512vbmi2>) noexcept
        {
            return _mm512_maskz_expand_epi16(mask.mask(), self);
        }
        template <class A>
        XSIMD_INLINE batch<uint16_t, A> expand(batch<uint16_t, A> const& self, batch_bool<uint16_t, A> const& mask, requires_arch<avx512vbmi2>) noexcept
        {
            return _mm512_maskz_expand_epi16(mask.mask(), self);
        }
        template <class A>
        XSIMD_INLINE batch<int8_t, A> expand(batch<int8_t, A> const& self, batch_bool<int8_t, A> const& mask, requires_arch<avx512vbmi2>) noexcept
        {
            return _mm512_maskz_expand_epi8(mask.mask(), self);
        }
        template <class A>
        XSIMD_INLINE batch<uint8_t, A> expand(batch<uint8_t, A> const& self, batch_bool<uint8_t, A> const& mask, requires_arch<avx512vbmi2>) noexcept
        {
            return _mm512_maskz_expand_epi8(mask.mask(), self);
        }
    }
}

#endif
