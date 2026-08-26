/****************************************************************************
 * Copyright (c) xsimd contributors                                         *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ARITHMETIC_X86_HPP
#define XSIMD_ARITHMETIC_X86_HPP

#include "../../arch/utils/x86.hpp"
#include "../../config/xsimd_macros.hpp"
#include "../../types/xsimd_batch.hpp"
#include "../../utils/xsimd_type_traits.hpp"
#include "../kernel_fwd.hpp"

#include <type_traits>

namespace xsimd::kernel
{
    namespace detail
    {
        /// A explicit trap with template so that type and architecture show in the error.
        template <class T, class A>
        constexpr void unsupported()
        {
            // static_assert(false) in a discarded if constexpr branch is only well-formed
            // since C++23 (P2593).
            static_assert(!std::is_same_v<A, A>, "unsupported data type for the given x86 architecture");
        }

        template <class T, class A>
        XSIMD_INLINE batch<T, A> mm_add(batch<T, A> lhs, batch<T, A> rhs) noexcept
        {
            if constexpr (std::is_same_v<T, float>)
            {
                return _mm_add_ps(lhs, rhs);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return _mm_add_pd(lhs, rhs);
            }
            else if constexpr (std::is_integral_v<T> && sizeof(T) == 1)
            {
                return _mm_add_epi8(lhs, rhs);
            }
            else if constexpr (std::is_integral_v<T> && sizeof(T) == 2)
            {
                return _mm_add_epi16(lhs, rhs);
            }
            else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
            {
                return _mm_add_epi32(lhs, rhs);
            }
            else if constexpr (std::is_integral_v<T> && sizeof(T) == 8)
            {
                return _mm_add_epi64(lhs, rhs);
            }
            else
            {
                unsupported<T, A>();
            }
        }

        template <class T, class A>
        XSIMD_INLINE batch<T, A> mm256_add(batch<T, A> lhs, batch<T, A> rhs) noexcept
        {
            if constexpr (std::is_same_v<T, float>)
            {
                return _mm256_add_ps(lhs, rhs);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return _mm256_add_pd(lhs, rhs);
            }
            else if constexpr (std::is_integral_v<T> && std::is_base_of_v<avx2, A>)
            {

                if constexpr (sizeof(T) == 1)
                {
                    return _mm256_add_epi8(lhs, rhs);
                }
                else if constexpr (sizeof(T) == 2)
                {
                    return _mm256_add_epi16(lhs, rhs);
                }
                else if constexpr (sizeof(T) == 4)
                {
                    return _mm256_add_epi32(lhs, rhs);
                }
                else if constexpr (sizeof(T) == 8)
                {
                    return _mm256_add_epi64(lhs, rhs);
                }
                else
                {
                    unsupported<T, A>();
                }
            }
            else
            {
                unsupported<T, A>();
            }
        }

        template <class T, class A>
        XSIMD_INLINE batch<T, A> mm512_add(batch<T, A> lhs, batch<T, A> rhs) noexcept
        {
            if constexpr (std::is_same_v<T, float>)
            {
                return _mm512_add_ps(lhs, rhs);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return _mm512_add_pd(lhs, rhs);
            }
            else if constexpr (std::is_integral_v<T> && sizeof(T) == 8)
            {
                return _mm512_add_epi64(lhs, rhs);
            }
            else if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
            {
                return _mm512_add_epi32(lhs, rhs);
            }
            else if constexpr (std::is_integral_v<T> && std::is_base_of_v<avx512bw, A>)
            {
                if constexpr (sizeof(T) == 1)
                {
                    return _mm512_add_epi8(lhs, rhs);
                }
                else if constexpr (sizeof(T) == 2)
                {
                    return _mm512_add_epi16(lhs, rhs);
                }
                else
                {
                    unsupported<T, A>();
                }
            }
            else
            {
                unsupported<T, A>();
            }
        }
    }

    template <class T, class A>
    XSIMD_INLINE batch<T, A> add(batch<T, A> lhs, batch<T, A> rhs) noexcept
    {
        constexpr auto recurse_add = [](auto l, auto r)
        { return kernel::add(l, r); };

        if constexpr (std::is_base_of_v<avx512f, A>)
        {
            if constexpr (!std::is_base_of_v<avx512bw, A> && std::is_integral_v<T> && sizeof(T) <= 2)
            {
                return detail::apply_on_halves(recurse_add, lhs, rhs);
            }
            else
            {
                return detail::mm512_add(lhs, rhs);
            }
        }
        else if constexpr (std::is_base_of_v<avx, A>)
        {
            if constexpr (!std::is_base_of_v<avx2, A> && std::is_integral_v<T>)
            {
                return detail::apply_on_halves(recurse_add, lhs, rhs);
            }
            else
            {
                return detail::mm256_add(lhs, rhs);
            }
        }
        else if constexpr (std::is_base_of_v<sse2, A>) // SSE family and avx<N>_128
        {
            return detail::mm_add(lhs, rhs);
        }
        else
        {
            detail::unsupported<T, A>();
        }
    }
}

#endif
