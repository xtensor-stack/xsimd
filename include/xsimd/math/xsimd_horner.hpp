/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_HORNER_HPP
#define XSIMD_HORNER_HPP

#include "../types/xsimd_types_include.hpp"

namespace xsimd
{

    namespace detail
    {
        template <class T, uint64_t c>
        inline T coef() noexcept
        {
            using value_type = typename T::value_type;
            return T(caster_t<value_type>(as_unsigned_integer_t<value_type>(c)).f);
        }
    }

    /**********
     * horner *
     **********/

    /* origin: boost/simdfunction/horn.hpp*/
    /*
     * ====================================================
     * copyright 2016 NumScale SAS
     *
     * Distributed under the Boost Software License, Version 1.0.
     * (See copy at http://boost.org/LICENSE_1_0.txt)
     * ====================================================
     */

    template <class T>
    inline T horner(const T&) noexcept
    {
        return T(0.);
    }

    template <class T, uint64_t c0>
    inline T horner(const T&) noexcept
    {
        return detail::coef<T, c0>();
    }

    template <class T, uint64_t c0, uint64_t c1, uint64_t... args>
    inline T horner(const T& x) noexcept
    {
        return fma(x, horner<T, c1, args...>(x), detail::coef<T, c0>());
    }

    /***********
     * horner1 *
     ***********/

    /* origin: boost/simdfunction/horn1.hpp*/
    /*
     * ====================================================
     * copyright 2016 NumScale SAS
     *
     * Distributed under the Boost Software License, Version 1.0.
     * (See copy at http://boost.org/LICENSE_1_0.txt)
     * ====================================================
     */

    template <class T>
    inline T horner1(const T&) noexcept
    {
        return T(1.);
    }

    template <class T, uint64_t c0>
    inline T horner1(const T& x) noexcept
    {
        return x + detail::coef<T, c0>();
    }

    template <class T, uint64_t c0, uint64_t c1, uint64_t... args>
    inline T horner1(const T& x) noexcept
    {
        return fma(x, horner1<T, c1, args...>(x), detail::coef<T, c0>());
    }
}

#endif
