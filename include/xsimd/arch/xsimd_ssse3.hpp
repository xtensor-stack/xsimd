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

#ifndef XSIMD_SSSE3_HPP
#define XSIMD_SSSE3_HPP

#include <cstddef>
#include <type_traits>

#include "../types/xsimd_ssse3_register.hpp"
#include "../types/xsimd_utils.hpp"

namespace xsimd
{

    namespace kernel
    {
        using namespace types;

        // abs
        template <class A, class T, typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, void>::type>
        inline batch<T, A> abs(batch<T, A> const& self, requires_arch<ssse3>) noexcept
        {
            switch (sizeof(T))
            {
            case 1:
                return _mm_abs_epi8(self);
            case 2:
                return _mm_abs_epi16(self);
            case 4:
                return _mm_abs_epi32(self);
            case 8:
                return _mm_abs_epi64(self);
            default:
                assert(false && "unsupported arch/op combination");
                return {};
            }
        }

        // extract_pair
        namespace detail
        {

            template <class T, class A>
            inline batch<T, A> extract_pair(batch<T, A> const&, batch<T, A> const& other, std::size_t, ::xsimd::detail::index_sequence<>) noexcept
            {
                return other;
            }

            template <class T, class A, std::size_t I, std::size_t... Is>
            inline batch<T, A> extract_pair(batch<T, A> const& self, batch<T, A> const& other, std::size_t i, ::xsimd::detail::index_sequence<I, Is...>) noexcept
            {
                if (i == I)
                {
                    return _mm_alignr_epi8(self, other, sizeof(T) * I);
                }
                else
                    return extract_pair(self, other, i, ::xsimd::detail::index_sequence<Is...>());
            }
        }

        template <class A, class T, class _ = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> extract_pair(batch<T, A> const& self, batch<T, A> const& other, std::size_t i, requires_arch<ssse3>) noexcept
        {
            constexpr std::size_t size = batch<T, A>::size;
            assert(0 <= i && i < size && "index in bounds");
            return detail::extract_pair(self, other, i, ::xsimd::detail::make_index_sequence<size>());
        }

        // hadd
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline T hadd(batch<T, A> const& self, requires_arch<ssse3>) noexcept
        {
            switch (sizeof(T))
            {
            case 2:
            {
                __m128i tmp1 = _mm_hadd_epi16(self, self);
                __m128i tmp2 = _mm_hadd_epi16(tmp1, tmp1);
                __m128i tmp3 = _mm_hadd_epi16(tmp2, tmp2);
                return _mm_cvtsi128_si32(tmp3) & 0xFFFF;
            }
            case 4:
            {
                __m128i tmp1 = _mm_hadd_epi32(self, self);
                __m128i tmp2 = _mm_hadd_epi32(tmp1, tmp1);
                return _mm_cvtsi128_si32(tmp2);
            }
            default:
                return hadd(self, sse3 {});
            }
        }
    }

}

#endif
