/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_CONVERSION_HPP
#define XSIMD_AVX512_CONVERSION_HPP

#include "xsimd_avx512_double.hpp"
#include "xsimd_avx512_float.hpp"
#include "xsimd_avx512_int32.hpp"
#include "xsimd_avx512_int64.hpp"
#include "xsimd_avx512_int16.hpp"
#include "xsimd_avx512_int8.hpp"

namespace xsimd
{

    /************************
     * conversion functions *
     ************************/

    batch<int32_t, 16> to_int(const batch<float, 16>& x);
    batch<int64_t, 8> to_int(const batch<double, 16>& x);

    batch<float, 16> to_float(const batch<int32_t, 16>& x);
    batch<double, 8> to_float(const batch<int64_t, 8>& x);

    /**************************
     * boolean cast functions *
     **************************/

    batch_bool<int32_t, 16> bool_cast(const batch_bool<float, 16>& x);
    batch_bool<int64_t, 8> bool_cast(const batch_bool<double, 8>& x);
    batch_bool<float, 16> bool_cast(const batch_bool<int32_t, 16>& x);
    batch_bool<double, 8> bool_cast(const batch_bool<int64_t, 8>& x);

    /*******************************
     * bitwise_cast implementation *
     *******************************/

    XSIMD_DEFINE_BITWISE_CAST_ALL(8)

    /***************************************
     * conversion functions implementation *
     ***************************************/

    inline batch<int32_t, 16> to_int(const batch<float, 16>& x)
    {
        return _mm512_cvttps_epi32(x);
    }

    inline batch<int64_t, 8> to_int(const batch<double, 8>& x)
    {
        return _mm512_cvtepi32_epi64(_mm512_cvttpd_epi32(x));
    }

    inline batch<float, 16> to_float(const batch<int32_t, 16>& x)
    {
        return _mm512_cvtepi32_ps(x);
    }

    inline batch<double, 8> to_float(const batch<int64_t, 8>& x)
    {
        return _mm512_cvtepi64_pd(x);
    }

    /**************************
     * boolean cast functions *
     **************************/

    inline batch_bool<int32_t, 16> bool_cast(const batch_bool<float, 16>& x)
    {
        return __mmask16(x);
    }

    inline batch_bool<int64_t, 8> bool_cast(const batch_bool<double, 8>& x)
    {
        return __mmask8(x);
    }

    inline batch_bool<float, 16> bool_cast(const batch_bool<int32_t, 16>& x)
    {
        return __mmask16(x);
    }

    inline batch_bool<double, 8> bool_cast(const batch_bool<int64_t, 8>& x)
    {
        return __mmask8(x);
    }

    /*****************************************
     * bitwise cast functions implementation *
     *****************************************/

    XSIMD_BITWISE_CAST_INTRINSIC(float, 16,
                                 double, 8,
                                 _mm512_castps_pd)

    XSIMD_BITWISE_CAST_INTRINSIC(float, 16,
                                 int32_t, 16,
                                 _mm512_castps_si512)

    XSIMD_BITWISE_CAST_INTRINSIC(float, 16,
                                 int64_t, 8,
                                 _mm512_castps_si512)

    XSIMD_BITWISE_CAST_INTRINSIC(double, 8,
                                 float, 16,
                                 _mm512_castpd_ps)

    XSIMD_BITWISE_CAST_INTRINSIC(double, 8,
                                 int32_t, 16,
                                 _mm512_castpd_si512)

    XSIMD_BITWISE_CAST_INTRINSIC(double, 8,
                                 int64_t, 8,
                                 _mm512_castpd_si512)

    XSIMD_BITWISE_CAST_INTRINSIC(int32_t, 16,
                                 float, 16,
                                 _mm512_castsi512_ps)

    XSIMD_BITWISE_CAST_INTRINSIC(int32_t, 16,
                                 double, 8,
                                 _mm512_castsi512_pd)

    XSIMD_BITWISE_CAST_INTRINSIC(int64_t, 8,
                                 float, 16,
                                 _mm512_castsi512_ps)

    XSIMD_BITWISE_CAST_INTRINSIC(int64_t, 8,
                                 double, 8,
                                 _mm512_castsi512_pd)
}

#endif
