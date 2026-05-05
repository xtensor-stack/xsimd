/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 * Copyright (c) Marco Barbone                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_AVX_128_HPP
#define XSIMD_AVX_128_HPP

#include <type_traits>

#include "../types/xsimd_avx_register.hpp"
#include "../types/xsimd_batch_constant.hpp"

namespace xsimd
{
    namespace kernel
    {
        using namespace types;

        // broadcast
        template <class A, class T, class = std::enable_if_t<std::is_same<T, float>::value>>
        XSIMD_INLINE batch<T, A> broadcast(T val, requires_arch<avx_128>) noexcept
        {
            return _mm_broadcast_ss(&val);
        }

        // eq
        template <class A>
        XSIMD_INLINE batch_bool<float, A> eq(batch<float, A> const& self, batch<float, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_ps(self, other, _CMP_EQ_OQ);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> eq(batch<double, A> const& self, batch<double, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_pd(self, other, _CMP_EQ_OQ);
        }

        // gt
        template <class A>
        XSIMD_INLINE batch_bool<float, A> gt(batch<float, A> const& self, batch<float, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_ps(self, other, _CMP_GT_OQ);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> gt(batch<double, A> const& self, batch<double, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_pd(self, other, _CMP_GT_OQ);
        }

        // ge
        template <class A>
        XSIMD_INLINE batch_bool<float, A> ge(batch<float, A> const& self, batch<float, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_ps(self, other, _CMP_GE_OQ);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> ge(batch<double, A> const& self, batch<double, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_pd(self, other, _CMP_GE_OQ);
        }

        // lt
        template <class A>
        XSIMD_INLINE batch_bool<float, A> lt(batch<float, A> const& self, batch<float, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_ps(self, other, _CMP_LT_OQ);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> lt(batch<double, A> const& self, batch<double, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_pd(self, other, _CMP_LT_OQ);
        }

        // le
        template <class A>
        XSIMD_INLINE batch_bool<float, A> le(batch<float, A> const& self, batch<float, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_ps(self, other, _CMP_LE_OQ);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> le(batch<double, A> const& self, batch<double, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_pd(self, other, _CMP_LE_OQ);
        }

        // neq
        template <class A>
        XSIMD_INLINE batch_bool<float, A> neq(batch<float, A> const& self, batch<float, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_ps(self, other, _CMP_NEQ_UQ);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> neq(batch<double, A> const& self, batch<double, A> const& other, requires_arch<avx_128>) noexcept
        {
            return _mm_cmp_pd(self, other, _CMP_NEQ_UQ);
        }

        // load_masked
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<float, A> load_masked(float const* mem, batch_bool_constant<float, A, Values...> mask, convert<float>, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskload_ps(mem, mask.as_batch());
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<double, A> load_masked(double const* mem, batch_bool_constant<double, A, Values...> mask, convert<double>, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskload_pd(mem, mask.as_batch());
        }

        // Runtime-mask load (float/double).
        template <class A, class Mode>
        XSIMD_INLINE batch<float, A>
        load_masked(float const* mem, batch_bool<float, A> mask, convert<float>, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskload_ps(mem, _mm_castps_si128(mask));
        }
        template <class A, class Mode>
        XSIMD_INLINE batch<double, A>
        load_masked(double const* mem, batch_bool<double, A> mask, convert<double>, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskload_pd(mem, _mm_castpd_si128(mask));
        }

        // store_masked
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(float* mem, batch<float, A> const& src, batch_bool_constant<float, A, Values...> mask, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskstore_ps(mem, mask.as_batch(), src);
        }

        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(double* mem, batch<double, A> const& src, batch_bool_constant<double, A, Values...> mask, Mode, requires_arch<avx_128>) noexcept
        {
            return _mm_maskstore_pd(mem, mask.as_batch(), src);
        }

        // Runtime-mask store (float/double).
        template <class A, class Mode>
        XSIMD_INLINE void
        store_masked(float* mem, batch<float, A> const& src, batch_bool<float, A> mask, Mode, requires_arch<avx_128>) noexcept
        {
            _mm_maskstore_ps(mem, _mm_castps_si128(mask), src);
        }
        template <class A, class Mode>
        XSIMD_INLINE void
        store_masked(double* mem, batch<double, A> const& src, batch_bool<double, A> mask, Mode, requires_arch<avx_128>) noexcept
        {
            _mm_maskstore_pd(mem, _mm_castpd_si128(mask), src);
        }

        // swizzle (dynamic mask)
        template <class A, class T, class ITy, class = std::enable_if_t<std::is_floating_point<T>::value && sizeof(T) == sizeof(ITy)>>
        XSIMD_INLINE batch<T, A> swizzle(batch<T, A> const& self, batch<ITy, A> mask, requires_arch<avx_128>) noexcept
        {
            XSIMD_IF_CONSTEXPR(std::is_same<T, float>::value)
            {
                return _mm_permutevar_ps(self, mask);
            }
            else
            {
                // VPERMILPD's variable control reads bit 1 of each 64-bit selector
                // (bit 0 is ignored), so a {0,1} index needs to become {0,2}.
                // Negation is a cheap alternative to a left shift by 1.
                return _mm_permutevar_pd(self, -mask);
            }
        }

        // swizzle (constant mask)
        template <class A, uint32_t V0, uint32_t V1, uint32_t V2, uint32_t V3>
        XSIMD_INLINE batch<float, A> swizzle(batch<float, A> const& self, batch_constant<uint32_t, A, V0, V1, V2, V3>, requires_arch<avx_128>) noexcept
        {
            return _mm_permute_ps(self, detail::mod_shuffle(V0, V1, V2, V3));
        }

        template <class A, uint32_t V0, uint32_t V1>
        XSIMD_INLINE batch<double, A> swizzle(batch<double, A> const& self, batch_constant<uint64_t, A, V0, V1>, requires_arch<avx_128>) noexcept
        {
            return _mm_permute_pd(self, detail::mod_shuffle(V0, V1));
        }

    }
}

#endif
