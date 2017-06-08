/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_POWER_HPP
#define XSIMD_POWER_HPP

namespace xsimd
{

    template <class T, std::size_t N>
    batch<T, N> hypot(const batch<T, N>& x, const batch<T, N>& y);

    /************************
     * hypot implementation *
     ************************/

    template <class T, std::size_t N>
    inline batch<T, N> hypot(const batch<T, N>& x, const batch<T, N>& y)
    {
        return sqrt(fma(x, x, y * y));
    }
}

#endif
