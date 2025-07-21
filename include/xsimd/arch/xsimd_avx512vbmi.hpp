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

#ifndef XSIMD_AVX512VBMI_HPP
#define XSIMD_AVX512VBMI_HPP

#include <array>
#include <type_traits>

#include "../types/xsimd_avx512vbmi_register.hpp"

namespace xsimd
{

    namespace kernel
    {
        using namespace types;

        namespace detail
        {
            template <size_t N>
            struct make_slide_left_bytes_pattern
            {
                static constexpr uint8_t get(size_t i, size_t)
                {
                    return i >= N ? i - N : 0;
                }
            };

            template <size_t N>
            struct make_slide_right_bytes_pattern
            {
                static constexpr uint8_t get(size_t i, size_t n)
                {
                    return i < (n - N) ? i + N : 0;
                }
            };
        }

        // slide_left
        template <size_t N, class A, class T>
        XSIMD_INLINE batch<T, A> slide_left(batch<T, A> const& x, requires_arch<avx512vbmi>) noexcept
        {
            if (N == 0)
            {
                return x;
            }
            if (N >= 64)
            {
                return batch<T, A>(T(0));
            }

            __mmask64 mask = 0xFFFFFFFFFFFFFFFFull << (N & 63);
            auto slide_pattern = xsimd::make_batch_constant<uint8_t, detail::make_slide_left_bytes_pattern<N>, A>();
            return _mm512_maskz_permutexvar_epi8(mask, slide_pattern.as_batch(), x);
        }

        // slide_right
        template <size_t N, class A, class T>
        XSIMD_INLINE batch<T, A> slide_right(batch<T, A> const& x, requires_arch<avx512vbmi>) noexcept
        {
            if (N == 0)
            {
                return x;
            }
            if (N >= 64)
            {
                return batch<T, A>(T(0));
            }
            __mmask64 mask = 0xFFFFFFFFFFFFFFFFull >> (N & 63);
            auto slide_pattern = xsimd::make_batch_constant<uint8_t, detail::make_slide_right_bytes_pattern<N>, A>();
            return _mm512_maskz_permutexvar_epi8(mask, slide_pattern.as_batch(), x);
        }

        // swizzle (dynamic version)
        template <class A>
        XSIMD_INLINE batch<uint8_t, A> swizzle(batch<uint8_t, A> const& self, batch<uint8_t, A> mask, requires_arch<avx512vbmi>) noexcept
        {
            return _mm512_permutexvar_epi8(mask, self);
        }

        template <class A>
        XSIMD_INLINE batch<int8_t, A> swizzle(batch<int8_t, A> const& self, batch<uint8_t, A> mask, requires_arch<avx512vbmi>) noexcept
        {
            return bitwise_cast<int8_t>(swizzle(bitwise_cast<uint8_t>(self), mask, avx512vbmi {}));
        }

        // swizzle (static version)
        template <class A, uint8_t... Vs>
        XSIMD_INLINE batch<uint8_t, A> swizzle(batch<uint8_t, A> const& self, batch_constant<uint8_t, A, Vs...> mask, requires_arch<avx512vbmi>) noexcept
        {
            return swizzle(self, mask.as_batch(), avx512vbmi {});
        }

        template <class A, uint8_t... Vs>
        XSIMD_INLINE batch<int8_t, A> swizzle(batch<int8_t, A> const& self, batch_constant<uint8_t, A, Vs...> mask, requires_arch<avx512vbmi>) noexcept
        {
            return swizzle(self, mask.as_batch(), avx512vbmi {});
        }
    }
}

#endif
