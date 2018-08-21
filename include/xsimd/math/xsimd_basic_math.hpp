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

    /**
     * @brief Computes the floating-point remainder of the division operation \c x/y.
     *
     * The floating-point remainder of the division operation \c x/y calculated by this
     * function is exactly the value <tt>x - n*y</tt>, where \c n is \c x/y with its fractional
     * part truncated. The returned value has the same sign as \c x and is less than \c y in magnitude.
     * @param x batch of floating point values.
     * @param y batch of floating point values.
     * @return the floating-point remainder of the division.
     */
    template <class T, std::size_t N>
    inline batch<T, N> fmod(const batch<T, N>& x, const batch<T, N>& y)
    {
        return fnma(trunc(x / y), y, x);
    }

    /**
     * @brief Computes the IEEE remainder of the floating point division operation \c x/y.
     *
     * The IEEE floating-point remainder of the division operation \c x/y calculated by this
     * function is exactly the value <tt>x - n*y</tt>, where the value n is the integral value
     * nearest the exact value \c x/y. When <tt>|n-x/y| = 0.5</tt>, the value n is chosen to be even.
     * In contrast to fmod, the returned value is not guaranteed to have the same sign as \c x.
     * If the returned value is 0, it will have the same sign as \c x.
     * @param x batch of floating point values.
     * @param y batch of floating point values.
     * @return the IEEE remainder remainder of the floating point division.
     */
    template <class T, std::size_t N>
    inline batch<T, N> remainder(const batch<T, N>& x, const batch<T, N>& y)
    {
        return fnma(nearbyint(x / y), y, x);
    }

    /**
     * Computes the positive difference between \c x and \c y, that is,
     * <tt>max(0, x-y)</tt>.
     * @param x batch of floating point values.
     * @param y batch of floating point values.
     * @return the positive difference.
     */
    template <class T, std::size_t N>
    inline batch<T, N> fdim(const batch<T, N>& x, const batch<T, N>& y)
    {
        return fmax(batch<T, N>(0.), x - y);
    }

    /**
     * Clips the values of the batch \c x between those of the batches \c lo and \c hi.
     * @param x batch of floating point values.
     * @param lo batch of floating point values.
     * @param hi batch of floating point values.
     * @return the result of the clipping.
     */
    template <class T, std::size_t N>
    inline batch<T, N> clip(const batch<T, N>& x, const batch<T, N>& lo, const batch<T, N>& hi)
    {
        return min(hi, max(x, lo));
    }
    template <class T>
    inline T clip(const T& x, const T& lo, const T& hi)
    {
        return std::min(hi, std::max(x, lo));
    }

    /*******************************************
     * Classification functions implementation *
     *******************************************/

    /**
     * Determines if the scalars in the given batch \c x have finite values,
     * i.e. theyr are different from infinite or NaN.
     * @param x batch of floating point values.
     * @return a batch of booleans.
     */
    template <class T, std::size_t N>
    inline batch_bool<T, N> isfinite(const batch<T, N>& x)
    {
        return (x - x) == batch<T, N>(0.);
    }

    /**
     * Determines if the scalars in the given batch \c x are positive
     * or negative infinity.
     * @param x batch of floating point values.
     * @return a batch of booleans.
     */
    template <class T, std::size_t N>
    inline batch_bool<T, N> isinf(const batch<T, N>& x)
    {
        return abs(x) == infinity<batch<T, N>>();
    }

    template <class T, std::size_t N>
    inline batch_bool<T, N> is_flint(const batch<T, N>& x)
    {
        using b_type = batch<T, N>;
        b_type frac = select(isnan(x - x), nan<b_type>(), x - trunc(x));
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
