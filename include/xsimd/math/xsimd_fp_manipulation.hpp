/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
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

    template <class T, std::size_t N>
    batch_bool<T, N> isfinite(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch_bool<T, N> isinf(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch_bool<T, N> is_flint(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch_bool<T, N> is_odd(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch_bool<T, N> is_even(const batch<T, N>& x);

    /**************************
     * Generic implementation *
     **************************/

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
        exp = select(bool_cast(arg != b_type(0.)), exp, i_type(0));
        return select((arg != b_type(0.)), x | bitwise_cast<b_type>(mask2frexp<b_type>()), b_type(0.));
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> isfinite(const batch<T, N>& x)
    {
        return (x - x) == batch<T, N>(0.);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> isinf(const batch<T, N>& x)
    {
        return abs(x) == infinity<batch<T, N>>();
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> is_flint(const batch<T, N>& x)
    {
        using b_type = batch<T, N>;
        b_type frac = select(is_nan(x - x), nan<b_type>(), x - trunc(x));
        return frac == b_type(0.);
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> is_odd(const batch<T, N>& x)
    {
        return is_even(x - batch<T, N>(1.));
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> is_even(const batch<T, N>& x)
    {
        return is_flint(x * batch<T, N>(0.5));
    }
}

#endif
