/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_SSE_CONVERSION_HPP
#define XSIMD_SSE_CONVERSION_HPP

#include "xsimd_sse_float.hpp"
#include "xsimd_sse_double.hpp"
#include "xsimd_sse_int.hpp"

namespace xsimd
{
    batch<int, 4> to_int(const batch<float, 4>& x);
    batch<int, 4> to_int(const batch<double, 2>& x);

    batch<float, 4> to_float(const batch<int, 4>& x);

    /**************************************
     * conversion function implementation *
     **************************************/

    inline batch<int, 4> to_int(const batch<float, 4>& x)
    {
        return _mm_cvttps_epi32(x);
    }

    inline batch<int, 4> to_int(const batch<double, 2>& x)
    {
        using batch_int = batch<int, 4>;
        __m128i tmp = _mm_cvttpd_epi32(x);
        return _mm_unpacklo_epi32(tmp, batch_int(tmp) < batch_int(0));
    }

    inline batch<float, 4> to_float(const batch<int, 4>& x)
    {
        return _mm_cvtepi32_ps(x);
    }
}

#endif

