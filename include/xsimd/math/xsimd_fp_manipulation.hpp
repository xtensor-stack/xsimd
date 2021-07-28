/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_FP_MANIPULATION_HPP
#define XSIMD_FP_MANIPULATION_HPP

#include "xsimd_numerical_constant.hpp"

namespace xsimd
{

    template <class T, std::size_t N>
    batch<T, N> ldexp(const batch<T, N>& x, const batch<as_integer_t<T>, N>& e);

    template <class T, std::size_t N>
    batch<T, N> frexp(const batch<T, N>& arg, batch<as_integer_t<T>, N>& exp);

    template <typename T>
    as_float_t<T> fexp_to_float(const T& x);

    /********************************************************
     * Floating point manipulation functions implementation *
     ********************************************************/

    /* origin: boost/simd/arch/common/simd/function/ldexp.hpp */
    /*
      * ====================================================
      * copyright 2016 NumScale SAS
      *
      * Distributed under the Boost Software License, Version 1.0.
      * (See copy at http://boost.org/LICENSE_1_0.txt)
      * ====================================================
      */
    template <class T, std::size_t N>
    inline batch<T, N> ldexp(const batch<T, N>& x, const batch<as_integer_t<T>, N>& e)
    {
        using btype = batch<T, N>;
        using itype = as_integer_t<btype>;
        itype ik = e + maxexponent<T>();
        ik = ik << nmb<T>();
        return x * bitwise_cast<btype>(ik);
    }

    /* origin: boost/simd/arch/common/simd/function/ifrexp.hpp */
    /*
     * ====================================================
     * copyright 2016 NumScale SAS
     *
     * Distributed under the Boost Software License, Version 1.0.
     * (See copy at http://boost.org/LICENSE_1_0.txt)
     * ====================================================
     */
    template <class T, std::size_t N>
    inline batch<T, N> frexp(const batch<T, N>& arg, batch<as_integer_t<T>, N>& exp)
    {
        using b_type = batch<T, N>;
        using i_type = batch<as_integer_t<T>, N>;
        i_type m1f = mask1frexp<b_type>();
        i_type r1 = m1f & bitwise_cast<i_type>(arg);
        b_type x = arg & bitwise_cast<b_type>(~m1f);
        exp = (r1 >> nmb<b_type>()) - maxexponentm1<b_type>();
        exp = select(bool_cast(arg != b_type(0.)), exp, zero<i_type>());
        return select((arg != b_type(0.)), x | bitwise_cast<b_type>(mask2frexp<b_type>()), b_type(0.));
    }

    template <typename T>
    inline as_float_t<T> fexp_to_float(const T& x)
    {
        return to_float(x);
    }

    // fexps are 8/11-bit numbers. so we can down-cast to int16_t/int32_t then do the job
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION && XSIMD_X86_INSTR_SET < XSIMD_X86_AVX512_VERSION
    inline batch<double, 4> fexp_to_float(const batch<int64_t, 4>& x)
    {
        return _mm256_cvtepi32_pd(detail::xsimd_cvtepi64_epi32(x));
    }
#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION && XSIMD_X86_INSTR_SET < XSIMD_X86_AVX512_VERSION
    inline batch<double, 2> fexp_to_float(const batch<int64_t, 2>& x)
    {
        return _mm_cvtepi32_pd(_mm_shuffle_epi32(x, _MM_SHUFFLE(0, 0, 2, 0)));
    }
#endif
}

#endif
