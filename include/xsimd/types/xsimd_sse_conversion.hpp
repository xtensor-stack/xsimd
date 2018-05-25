/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_SSE_CONVERSION_HPP
#define XSIMD_SSE_CONVERSION_HPP

#include "xsimd_sse_double.hpp"
#include "xsimd_sse_float.hpp"
#include "xsimd_sse_int32.hpp"
#include "xsimd_sse_int64.hpp"

namespace xsimd
{

    /************************
     * conversion functions *
     ************************/

    batch<int32_t, 4> to_int(const batch<float, 4>& x);
    batch<int64_t, 2> to_int(const batch<double, 2>& x);

    batch<float, 4> to_float(const batch<int32_t, 4>& x);
    batch<double, 2> to_float(const batch<int64_t, 2>& x);

    /**************************
     * boolean cast functions *
     **************************/

    batch_bool<int32_t, 4> bool_cast(const batch_bool<float, 4>& x);
    batch_bool<int64_t, 2> bool_cast(const batch_bool<double, 2>& x);
    batch_bool<float, 4> bool_cast(const batch_bool<int32_t, 4>& x);
    batch_bool<double, 2> bool_cast(const batch_bool<int64_t, 2>& x);

    /***************************************
     * conversion functions implementation *
     ***************************************/

    inline batch<int32_t, 4> to_int(const batch<float, 4>& x)
    {
        return _mm_cvttps_epi32(x);
    }

    inline batch<int64_t, 2> to_int(const batch<double, 2>& x)
    {
        using batch_int = batch<int64_t, 2>;
        __m128i tmp = _mm_cvttpd_epi32(x);
        return _mm_unpacklo_epi32(tmp, batch_int(tmp) < batch_int(int64_t(0)));
    }

    inline batch<float, 4> to_float(const batch<int32_t, 4>& x)
    {
        return _mm_cvtepi32_ps(x);
    }

    inline batch<double, 2> to_float(const batch<int64_t, 2>& x)
    {
        return batch<double, 2>(static_cast<double>(x[0]), static_cast<double>(x[1]));
    }

    /**************************
     * boolean cast functions *
     **************************/

    inline batch_bool<int32_t, 4> bool_cast(const batch_bool<float, 4>& x)
    {
        return _mm_castps_si128(x);
    }

    inline batch_bool<int64_t, 2> bool_cast(const batch_bool<double, 2>& x)
    {
        return _mm_castpd_si128(x);
    }

    inline batch_bool<float, 4> bool_cast(const batch_bool<int32_t, 4>& x)
    {
        return _mm_castsi128_ps(x);
    }

    inline batch_bool<double, 2> bool_cast(const batch_bool<int64_t, 2>& x)
    {
        return _mm_castsi128_pd(x);
    }

    /*****************************************
     * bitwise cast functions implementation *
     *****************************************/

    XSIMD_BITWISE_CAST_INTRINSIC(float, 4,
                                 double, 2,
                                 _mm_castps_pd)

    XSIMD_BITWISE_CAST_INTRINSIC(float, 4,
                                 int32_t, 4,
                                 _mm_castps_si128)

    XSIMD_BITWISE_CAST_INTRINSIC(float, 4,
                                 int64_t, 2,
                                 _mm_castps_si128)

    XSIMD_BITWISE_CAST_INTRINSIC(double, 2,
                                 float, 4,
                                 _mm_castpd_ps)

    XSIMD_BITWISE_CAST_INTRINSIC(double, 2,
                                 int32_t, 4,
                                 _mm_castpd_si128)

    XSIMD_BITWISE_CAST_INTRINSIC(double, 2,
                                 int64_t, 2,
                                 _mm_castpd_si128)

    XSIMD_BITWISE_CAST_INTRINSIC(int32_t, 4,
                                 float, 4,
                                 _mm_castsi128_ps)

    XSIMD_BITWISE_CAST_INTRINSIC(int32_t, 4,
                                 double, 2,
                                 _mm_castsi128_pd)

    XSIMD_BITWISE_CAST_INTRINSIC(int64_t, 2,
                                 float, 4,
                                 _mm_castsi128_ps)

    XSIMD_BITWISE_CAST_INTRINSIC(int64_t, 2,
                                 double, 2,
                                 _mm_castsi128_pd)
}

#endif
