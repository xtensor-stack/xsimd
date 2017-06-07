/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_FP_SIGN_HPP
#define XSIMD_FP_SIGN_HPP

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
 
    template <class T, std::size_t N>
    inline batch<T, N> signnz(const batch<T, N>& x)
    {
        using batch_type = batch<T, N>;
        return batch_type(1) | (signmask<batch_type>() & x);
    }
}

#endif

