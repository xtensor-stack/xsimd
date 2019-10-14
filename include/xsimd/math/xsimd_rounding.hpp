/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_ROUNDING_HPP
#define XSIMD_ROUNDING_HPP

#include <cmath>

#include "xsimd_fp_sign.hpp"
#include "xsimd_numerical_constant.hpp"

namespace xsimd
{
    /**
     * Computes the batch of smallest integer values not less than
     * scalars in \c x.
     * @param x batch of floating point values.
     * @return the batch of smallest integer values not less than \c x.
     */
    template <class B>
    batch_type_t<B> ceil(const simd_base<B>& x);

    /**
     * Computes the batch of largest integer values not greater than
     * scalars in \c x.
     * @param x batch of floating point values.
     * @return the batch of largest integer values not greater than \c x.
     */
    template <class B>
    batch_type_t<B> floor(const simd_base<B>& x);

    /**
     * Computes the batch of nearest integer values not greater in magnitude
     * than scalars in \c x.
     * @param x batch of floating point values.
     * @return the batch of nearest integer values not greater in magnitude than \c x.
     */
    template <class B>
    batch_type_t<B> trunc(const simd_base<B>& x);

    /**
     * Computes the batch of nearest integer values to scalars in \c x (in
     * floating point format), rounding halfway cases away from zero, regardless
     * of the current rounding mode.
     * @param x batch of flaoting point values.
     * @return the batch of nearest integer values. 
     */
    template <class B>
    batch_type_t<B> round(const simd_base<B>& x);

    // Contrary to their std counterpart, these functions
    // are assume that the rounding mode is FE_TONEAREST

    /**
     * Rounds the scalars in \c x to integer values (in floating point format), using
     * the current rounding mode.
     * @param x batch of flaoting point values.
     * @return the batch of nearest integer values.
     */
    template <class B>
    batch_type_t<B> nearbyint(const simd_base<B>& x);

    /**
     * Rounds the scalars in \c x to integer values (in floating point format), using
     * the current rounding mode.
     * @param x batch of flaoting point values.
     * @return the batch of rounded values.
     */
    template <class B>
    batch_type_t<B> rint(const simd_base<B>& x);

    namespace impl
    {
        template <class T, std::size_t N>
        batch<T, N> ceil(const batch<T, N>& x);

        template <class T, std::size_t N>
        batch<T, N> floor(const batch<T, N>& x);

        template <class T, std::size_t N>
        batch<T, N> trunc(const batch<T, N>& x);

        template <class T, std::size_t N>
        batch<T, N> round(const batch<T, N>& x);

        template <class T, std::size_t N>
        batch<T, N> nearbyint(const batch<T, N>& x);

        template <class T, std::size_t N>
        batch<T, N> rint(const batch<T, N>& x);

        /**********************
         * SSE implementation *
         **********************/

    #if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION

        template <>
        inline batch<float, 4> ceil(const batch<float, 4>& x)
        {
            return _mm_ceil_ps(x);
        }

        template <>
        inline batch<double, 2> ceil(const batch<double, 2>& x)
        {
            return _mm_ceil_pd(x);
        }

        template <>
        inline batch<float, 4> floor(const batch<float, 4>& x)
        {
            return _mm_floor_ps(x);
        }

        template <>
        inline batch<double, 2> floor(const batch<double, 2>& x)
        {
            return _mm_floor_pd(x);
        }

        template <>
        inline batch<float, 4> trunc(const batch<float, 4>& x)
        {
            return _mm_round_ps(x, _MM_FROUND_TO_ZERO);
        }

        template <>
        inline batch<double, 2> trunc(const batch<double, 2>& x)
        {
            return _mm_round_pd(x, _MM_FROUND_TO_ZERO);
        }

        template <>
        inline batch<float, 4> nearbyint(const batch<float, 4>& x)
        {
            return _mm_round_ps(x, _MM_FROUND_TO_NEAREST_INT);
        }

        template <>
        inline batch<double, 2> nearbyint(const batch<double, 2>& x)
        {
            return _mm_round_pd(x, _MM_FROUND_TO_NEAREST_INT);
        }

    #elif (XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION) || (XSIMD_ARM_INSTR_SET == XSIMD_ARM7_NEON_VERSION)

        // NOTE: Renamed to avoid collision with the fallback implementation
        template <class T, std::size_t N>
        inline batch<T, N> ceil_impl(const batch<T, N>& x)
        {
            using btype = batch<T, N>;
            btype tx = trunc(x);
            return select(tx < x, tx + btype(1), tx);
        }

        template <class T, std::size_t N>
        inline batch<T, N> floor_impl(const batch<T, N>& x)
        {
            using btype = batch<T, N>;
            btype tx = trunc(x);
            return select(tx > x, tx - btype(1), tx);
        }

        template <class T, std::size_t N>
        inline batch<T, N> nearbyint_impl(const batch<T, N>& x)
        {
            using btype = batch<T, N>;
            btype s = bitofsign(x);
            btype v = x ^ s;
            btype t2n = twotonmb<btype>();
            btype d0 = v + t2n;
            return s ^ select(v < t2n, d0 - t2n, v);
        }

        template <>
        inline batch<float, 4> ceil(const batch<float, 4>& x)
        {
            return ceil_impl(x);
        }

        template <>
        inline batch<float, 4> floor(const batch<float, 4>& x)
        {
            return floor_impl(x);
        }

        template <>
        inline batch<float, 4> nearbyint(const batch<float, 4>& x)
        {
            return nearbyint_impl(x);
        }

        template <>
        inline batch<float, 4> trunc(const batch<float, 4>& x)
        {
            using btype = batch<float, 4>;
            return select(abs(x) < maxflint<btype>(), to_float(to_int(x)), x);
        }

    #if (XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION)
        template <>
        inline batch<double, 2> ceil(const batch<double, 2>& x)
        {
            return ceil_impl(x);
        }

        template <>
        inline batch<double, 2> floor(const batch<double, 2>& x)
        {
            return floor_impl(x);
        }

        template <>
        inline batch<double, 2> nearbyint(const batch<double, 2>& x)
        {
            return nearbyint_impl(x);
        }

        template <>
        inline batch<double, 2> trunc(const batch<double, 2>& x)
        {
            return batch<double, 2>(std::trunc(x[0]), std::trunc(x[1]));
        }
    #endif
    #endif

        /**********************
         * AVX implementation *
         **********************/

    #if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION

        template <>
        inline batch<float, 8> ceil(const batch<float, 8>& x)
        {
            return _mm256_round_ps(x, _MM_FROUND_CEIL);
        }

        template <>
        inline batch<double, 4> ceil(const batch<double, 4>& x)
        {
            return _mm256_round_pd(x, _MM_FROUND_CEIL);
        }

        template <>
        inline batch<float, 8> floor(const batch<float, 8>& x)
        {
            return _mm256_round_ps(x, _MM_FROUND_FLOOR);
        }

        template <>
        inline batch<double, 4> floor(const batch<double, 4>& x)
        {
            return _mm256_round_pd(x, _MM_FROUND_FLOOR);
        }

        template <>
        inline batch<float, 8> trunc(const batch<float, 8>& x)
        {
            return _mm256_round_ps(x, _MM_FROUND_TO_ZERO);
        }

        template <>
        inline batch<double, 4> trunc(const batch<double, 4>& x)
        {
            return _mm256_round_pd(x, _MM_FROUND_TO_ZERO);
        }

        template <>
        inline batch<float, 8> nearbyint(const batch<float, 8>& x)
        {
            return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT);
        }

        template <>
        inline batch<double, 4> nearbyint(const batch<double, 4>& x)
        {
            return _mm256_round_pd(x, _MM_FROUND_TO_NEAREST_INT);
        }

    #endif

    #if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512_VERSION

        template <>
        inline batch<float, 16> ceil(const batch<float, 16>& x)
        {
            auto res = _mm512_ceil_ps(x);
            return res;
        }

        template <>
        inline batch<double, 8> ceil(const batch<double, 8>& x)
        {
            auto res = _mm512_ceil_pd(x);
            return res;
        }

        template <>
        inline batch<float, 16> floor(const batch<float, 16>& x)
        {
            auto res = _mm512_floor_ps(x);
            return res;
        }

        template <>
        inline batch<double, 8> floor(const batch<double, 8>& x)
        {
            auto res = _mm512_floor_pd(x);
            return res;
        }

        template <>
        inline batch<float, 16> trunc(const batch<float, 16>& x)
        {
            auto res = _mm512_roundscale_round_ps(x, _MM_FROUND_TO_ZERO, _MM_FROUND_CUR_DIRECTION);
            return res;
        }

        template <>
        inline batch<double, 8> trunc(const batch<double, 8>& x)
        {
            auto res = _mm512_roundscale_round_pd(x, _MM_FROUND_TO_ZERO, _MM_FROUND_CUR_DIRECTION);
            return res;
        }

        template <>
        inline batch<float, 16> nearbyint(const batch<float, 16>& x)
        {
            auto res = _mm512_roundscale_round_ps(x, _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_CUR_DIRECTION);
            return res;
        }

        template <>
        inline batch<double, 8> nearbyint(const batch<double, 8>& x)
        {
            auto res = _mm512_roundscale_round_pd(x, _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_CUR_DIRECTION);
            return res;
        }

    #endif

    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_32_NEON_VERSION
        template <>
        inline batch<float, 4> ceil(const batch<float, 4>& x)
        {
            return vrndpq_f32(x);
        }

        template <>
        inline batch<float, 4> floor(const batch<float, 4>& x)
        {
            return vrndmq_f32(x);
        }
        template <>
        inline batch<float, 4> trunc(const batch<float, 4>& x)
        {
            return vrndq_f32(x);
        }

        template <>
        inline batch<float, 4> nearbyint(const batch<float, 4>& x)
        {
            return vrndxq_f32(x);
        }
    #endif

    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        template <>
        inline batch<double, 2> ceil(const batch<double, 2>& x)
        {
            return vrndpq_f64(x);
        }

        template <>
        inline batch<double, 2> floor(const batch<double, 2>& x)
        {
            return vrndmq_f64(x);
        }

        template <>
        inline batch<double, 2> trunc(const batch<double, 2>& x)
        {
            return vrndq_f64(x);
        }

        template <>
        inline batch<double, 2> nearbyint(const batch<double, 2>& x)
        {
            return vrndxq_f64(x);
        }
    #endif

        /***************************
         * Fallback implementation *
         ***************************/

    #if defined(XSIMD_ENABLE_FALLBACK)
        template <class T, std::size_t N>
        inline batch<T, N> ceil(const batch<T, N>& x)
        {
            XSIMD_FALLBACK_BATCH_UNARY_FUNC(std::ceil, x)
        }

        template <class T, std::size_t N>
        inline batch<T, N> floor(const batch<T, N>& x)
        {
            XSIMD_FALLBACK_BATCH_UNARY_FUNC(std::floor, x)
        }

        template <class T, std::size_t N>
        inline batch<T, N> trunc(const batch<T, N>& x)
        {
            XSIMD_FALLBACK_BATCH_UNARY_FUNC(std::trunc, x)
        }

        template <class T, std::size_t N>
        inline batch<T, N> nearbyint(const batch<T, N>& x)
        {
            XSIMD_FALLBACK_BATCH_UNARY_FUNC(std::nearbyint, x)
        }
    #endif

        /**************************
         * Generic implementation *
         **************************/

        template <class T, std::size_t N>
        inline batch<T, N> round(const batch<T, N>& x)
        {
            using btype = batch<T, N>;
            btype v = abs(x);
            btype c = ceil(v);
            btype cp = select(c - btype(0.5) > v, c - btype(1), c);
            return select(v > maxflint<btype>(), x, copysign(cp, x));
        }

        template <class T, std::size_t N>
        inline batch<T, N> rint(const batch<T, N>& x)
        {
            return nearbyint(x);
        }
    }


    template <class B>
    inline batch_type_t<B> ceil(const simd_base<B>& x)
    {
        return impl::ceil(x());
    }

    template <class B>
    inline batch_type_t<B> floor(const simd_base<B>& x)
    {
        return impl::floor(x());
    }

    template <class B>
    inline batch_type_t<B> trunc(const simd_base<B>& x)
    {
        return impl::trunc(x());
    }

    template <class B>
    inline batch_type_t<B> round(const simd_base<B>& x)
    {
        return impl::round(x());
    }

    // Contrary to their std counterpart, these functions
    // are assume that the rounding mode is FE_TONEAREST
    template <class B>
    inline batch_type_t<B> nearbyint(const simd_base<B>& x)
    {
        return impl::nearbyint(x());
    }

    template <class B>
    inline batch_type_t<B> rint(const simd_base<B>& x)
    {
        return impl::rint(x());
    }

}

#endif
