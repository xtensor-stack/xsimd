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

    /**************************
     * Generic implementation *
     **************************/

    template <class T, std::size_t N>
    inline batch<T, N> ldexp(const batch<T, N>& x, const batch<as_integer_t<T>, N>& e)
    {
        using btype = batch<T, N>;
        using itype = batch<as_integer_t<T>, N>;
        itype ik = e + itype(maxexponent<T>());
        ik = ik << nmb<T>();
        return x * bitwise_cast<btype>(ik);
    }

}

#endif
