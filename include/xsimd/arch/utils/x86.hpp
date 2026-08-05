/****************************************************************************
 * Copyright (c) xsimd contributors                                         *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ARCH_UTILS_AVX_HPP
#define XSIMD_ARCH_UTILS_AVX_HPP

#include "../../config/xsimd_macros.hpp"
#include "../../types/xsimd_batch.hpp"
#include "../../types/xsimd_x86_registers.hpp"

#include <type_traits>

namespace xsimd::kernel::detail
{
    template <class T, class A>
    using half_batch_t = make_sized_batch_t<T, batch<T, A>::size / 2>;

    template <class T, class A>
    using half_arch_t = typename half_batch_t<T, A>::arch_type;

    template <class T, class A2, class A1 = half_arch_t<T, A2>>
    XSIMD_INLINE batch<T, A1> lower_half(batch<T, A2> self) noexcept
    {
        if constexpr (sizeof(self) == 64)
        {
            if constexpr (std::is_same_v<T, float>)
            {
                return _mm512_castps512_ps256(self);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return _mm512_castpd512_pd256(self);
            }
            else if constexpr (std::is_integral_v<T>)
            {
                return _mm512_castsi512_si256(self);
            }
        }
        else if constexpr (sizeof(self) == 32)
        {
            if constexpr (sizeof(self) == 32 && std::is_same_v<T, float>)
            {
                return _mm256_castps256_ps128(self);
            }
            else if constexpr (sizeof(self) == 32 && std::is_same_v<T, double>)
            {
                return _mm256_castpd256_pd128(self);
            }
            else if constexpr (sizeof(self) == 32 && std::is_integral_v<T>)
            {
                return _mm256_castsi256_si128(self);
            }
        }
        else
        {
            static_assert(false, "unsupported architecture conversion");
        }
    }

    template <class T, class A2, class A1 = half_arch_t<T, A2>>
    XSIMD_INLINE batch<T, A1> upper_half(batch<T, A2> self) noexcept
    {
        if constexpr (sizeof(self) == 64)
        {
            if constexpr (std::is_same_v<T, float>)
            {
                // _mm512_extractf32x8_ps is AVX512DQ but the casts here are a noop
                return _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(self), 1));
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return _mm512_extractf64x4_pd(self, 1);
            }
            else if constexpr (std::is_integral_v<T>)
            {
                return _mm512_extracti64x4_epi64(self, 1);
            }
        }
        else if constexpr (sizeof(self) == 32)
        {
            if constexpr (std::is_same_v<T, float>)
            {
                return _mm256_extractf128_ps(self, 1);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return _mm256_extractf128_pd(self, 1);
            }
            else if constexpr (std::is_integral_v<T>)
            {
                return _mm256_extractf128_si256(self, 1);
            }
        }
        else
        {
            static_assert(false, "unsupported architecture conversion");
        }
    }

    template <class T, class A2, class A1 = half_arch_t<T, A2>>
    XSIMD_INLINE batch<T, A2> merge_halves(batch<T, A1> low, batch<T, A1> high) noexcept
    {
        if constexpr (sizeof(batch<T, A2>) == 64)
        {
            if constexpr (std::is_same_v<T, float>)
            {
                // _mm512_insertf32x8 is AVX512DQ but the casts here are a noop
                auto const ld = _mm256_castps_pd(low);
                auto const lh = _mm256_castps_pd(high);
                return _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castpd256_pd512(ld), lh, 1));
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return _mm512_insertf64x4(_mm512_castpd256_pd512(low), high, 1);
            }
            else if constexpr (std::is_integral_v<T>)
            {
                return _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
            }
        }
        if constexpr (sizeof(batch<T, A2>) == 32)
        {
            if constexpr (std::is_same_v<T, float>)
            {
                return _mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 1);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return _mm256_insertf128_pd(_mm256_castpd128_pd256(low), high, 1);
            }
            else if constexpr (std::is_integral_v<T>)
            {
                return _mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1);
            }
        }
        else
        {
            static_assert(false, "unsupported architecture conversion");
        }
    }

    template <class A1, class T, class A2, class F>
    XSIMD_INLINE batch<T, A2> apply_on_halves_with_arch(F&& f, batch<T, A2> self) noexcept
    {
        auto low = f(lower_half<T, A2, A1>(self));
        auto high = f(upper_half<T, A2, A1>(self));
        return merge_halves<T, A2, A1>(low, high);
    }

    template <class A1, class T, class A2, class F>
    XSIMD_INLINE batch<T, A2> apply_on_halves_with_arch(F&& f, batch<T, A2> lhs, batch<T, A2> rhs) noexcept
    {
        auto low = f(lower_half<T, A2, A1>(lhs), lower_half<T, A2, A1>(rhs));
        auto high = f(upper_half<T, A2, A1>(lhs), upper_half<T, A2, A1>(rhs));
        return merge_halves<T, A2, A1>(low, high);
    }

    template <class T, class A2, class F>
    XSIMD_INLINE batch<T, A2> apply_on_halves(F&& f, batch<T, A2> self) noexcept
    {
        using A1 = half_arch_t<T, A2>;
        return apply_on_halves_with_arch<A1, T, A2, F>(std::forward<F>(f), self);
    }

    template <class T, class A2, class F>
    XSIMD_INLINE batch<T, A2> apply_on_halves(F&& f, batch<T, A2> lhs, batch<T, A2> rhs) noexcept
    {
        using A1 = half_arch_t<T, A2>;
        return apply_on_halves_with_arch<A1, T, A2, F>(std::forward<F>(f), lhs, rhs);
    }
}
#endif
