/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_BASIC_MATH_HPP
#define XSIMD_BASIC_MATH_HPP

#include "xsimd_numerical_constant.hpp"
#include "xsimd_rounding.hpp"

namespace xsimd
{
    /********************
     * Basic operations *
     ********************/

    template <class T, std::size_t N>
    batch<T, N> fmod(const batch<T, N>& x, const batch<T, N>& y);

    template <class T, std::size_t N>
    batch<T, N> remainder(const batch<T, N>& x, const batch<T, N>& y);

    template <class T, std::size_t N>
    batch<T, N> fdim(const batch<T, N>& x, const batch<T, N>& y);

    template <class T, std::size_t N>
    batch<T, N> clip(const batch<T, N>& x, const batch<T, N>& lo, const batch<T, N>& hi);

    /****************************
     * Classification functions *
     ****************************/

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

    /***********************************
     * Basic operations implementation *
     ***********************************/

    template <class T, std::size_t N>
    inline batch<T, N> fmod(const batch<T, N>& x, const batch<T, N>& y)
    {
        return fnma(trunc(x / y), y, x);
    }

    template <class T, std::size_t N>
    inline batch<T, N> remainder(const batch<T, N>& x, const batch<T, N>& y)
    {
        return fnma(nearbyint(x / y), y, x);
    }

    template <class T, std::size_t N>
    inline batch<T, N> fdim(const batch<T, N>& x, const batch<T, N>& y)
    {
        return fmax(batch<T, N>(0.), x - y);
    }

    template <class T, std::size_t N>
    inline batch<T, N> clip(const batch<T, N>& x, const batch<T, N>& lo, const batch<T, N>& hi)
    {
        return select(x < lo, lo, select(x > hi, hi, x));
    }

    /*******************************************
     * Classification functions implementation *
     *******************************************/

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
