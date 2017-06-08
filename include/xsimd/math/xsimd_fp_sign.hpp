/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_FP_SIGN_HPP
#define XSIMD_FP_SIGN_HPP

#include <type_traits>
#include "xsimd_numerical_constant.hpp"

namespace xsimd
{

    template <class T, std::size_t N>
    batch<T, N> bitofsign(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch<T, N> copysign(const batch<T, N>& x1, const batch<T, N>& x2);

    template <class T, std::size_t N>
    batch<T, N> signnz(const batch<T, N>& x);

    /**************************
     * fp_sign implementation *
     **************************/

    template <class T, std::size_t N>
    inline batch<T, N> bitofsign(const batch<T, N>& x)
    {
        using btype = batch<T, N>;
        return x & minuszero<btype>();
    }

    template <class T, std::size_t N>
    inline batch<T, N> copysign(const batch<T, N>& x1, const batch<T, N>& x2)
    {
        return abs(x1) | bitofsign(x2);
    }
 
    namespace detail
    {
        template <class B, bool cond = std::is_floating_point<typename B::value_type>::value>
        struct signnz_impl
        {
            static inline B compute(const B& x)
            {
                using value_type = typename B::value_type;
                return (x >> (sizeof(value_type) * 8 - 1)) | B(1.);
            }
        };

        template <class B>
        struct signnz_impl<B, true>
        {
            static inline B compute(const B& x)
            {
                return B(1.) | (signmask<B>() & x);
            }
        };
    }
    template <class T, std::size_t N>
    inline batch<T, N> signnz(const batch<T, N>& x)
    {
        return detail::signnz_impl<batch<T, N>>::compute(x);
    }
}

#endif

