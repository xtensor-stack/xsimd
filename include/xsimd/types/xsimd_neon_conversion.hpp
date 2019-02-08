/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_NEON_CONVERSION_HPP
#define XSIMD_NEON_CONVERSION_HPP

#include "xsimd_neon_bool.hpp"
#include "xsimd_neon_float.hpp"
#include "xsimd_neon_int32.hpp"
#include "xsimd_neon_int64.hpp"
#include "xsimd_neon_int16.hpp"
#include "xsimd_neon_int8.hpp"
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    #include "xsimd_neon_double.hpp"
#endif
#include "xsimd_neon_uint32.hpp"
#include "xsimd_neon_uint64.hpp"
#include "xsimd_neon_uint16.hpp"
#include "xsimd_neon_uint8.hpp"

namespace xsimd
{

    /************************
     * conversion functions *
     ************************/

    batch<int32_t, 4> to_int(const batch<float, 4>& x);
    batch<float, 4> to_float(const batch<int32_t, 4>& x);

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    batch<int64_t, 2> to_int(const batch<double, 2>& x);
    batch<double, 2> to_float(const batch<int64_t, 2>& x);
#endif

    /**************************
     * boolean cast functions *
     **************************/

    batch_bool<int32_t, 4> bool_cast(const batch_bool<float, 4>& x);
    batch_bool<float, 4> bool_cast(const batch_bool<int32_t, 4>& x);

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    batch_bool<int64_t, 2> bool_cast(const batch_bool<double, 2>& x);
    batch_bool<double, 2> bool_cast(const batch_bool<int64_t, 2>& x);
#endif

    /*******************************
     * bitwise_cast implementation *
     *******************************/

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    XSIMD_DEFINE_BITWISE_CAST_FLOAT(double, 2)
#endif
    XSIMD_DEFINE_BITWISE_CAST_FLOAT(float, 4)
    XSIMD_DEFINE_BITWISE_CAST(int64_t, 2)
    XSIMD_DEFINE_BITWISE_CAST(uint64_t, 2)
    XSIMD_DEFINE_BITWISE_CAST(int32_t, 4)
    XSIMD_DEFINE_BITWISE_CAST(uint32_t, 4)
    XSIMD_DEFINE_BITWISE_CAST(int16_t, 8)
    XSIMD_DEFINE_BITWISE_CAST(uint16_t, 8)
    XSIMD_DEFINE_BITWISE_CAST(int8_t, 16)
    XSIMD_DEFINE_BITWISE_CAST(uint8_t, 16)
    
    /***************************************
     * conversion functions implementation *
     ***************************************/

    inline batch<int32_t, 4> to_int(const batch<float, 4>& x)
    {
        return vcvtq_s32_f32(x);
    }

    inline batch<float, 4> to_float(const batch<int32_t, 4>& x)
    {
        return vcvtq_f32_s32(x);
    }

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    inline batch<int64_t, 2> to_int(const batch<double, 2>& x)
    {
        return vcvtq_s64_f64(x);
    }

    inline batch<double, 2> to_float(const batch<int64_t, 2>& x)
    {
        return vcvtq_f64_s64(x);
    }
#endif

    /**************************
     * boolean cast functions *
     **************************/

    inline batch_bool<int32_t, 4> bool_cast(const batch_bool<float, 4>& x)
    {
        return x;
    }

    inline batch_bool<float, 4> bool_cast(const batch_bool<int32_t, 4>& x)
    {
        return x;
    }

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    inline batch_bool<int64_t, 2> bool_cast(const batch_bool<double, 2>& x)
    {
        return x;
    }

    inline batch_bool<double, 2> bool_cast(const batch_bool<int64_t, 2>& x)
    {
        return x;
    }
#endif

    /*****************************************
     * bitwise cast functions implementation *
     *****************************************/

    XSIMD_BITWISE_CAST_INTRINSIC(float, 4,
                                 int32_t, 4,
                                 vreinterpretq_s32_f32)

    XSIMD_BITWISE_CAST_INTRINSIC(float, 4,
                                 int64_t, 2,
                                 vreinterpretq_s64_f32)

    XSIMD_BITWISE_CAST_INTRINSIC(int32_t, 4,
                                 float, 4,
                                 vreinterpretq_f32_s32)

    XSIMD_BITWISE_CAST_INTRINSIC(int64_t, 2,
                                 float, 4,
                                 vreinterpretq_f32_s64)

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    XSIMD_BITWISE_CAST_INTRINSIC(double, 2,
                                 float, 4,
                                 vreinterpretq_f32_f64)

    XSIMD_BITWISE_CAST_INTRINSIC(double, 2,
                                 int32_t, 4,
                                 vreinterpretq_s32_f64)

    XSIMD_BITWISE_CAST_INTRINSIC(double, 2,
                                 int64_t, 2,
                                 vreinterpretq_s64_f64)

    XSIMD_BITWISE_CAST_INTRINSIC(int32_t, 4,
                                 double, 2,
                                 vreinterpretq_f64_s32)

    XSIMD_BITWISE_CAST_INTRINSIC(int64_t, 2,
                                 double, 2,
                                 vreinterpretq_f64_s64)

    XSIMD_BITWISE_CAST_INTRINSIC(float, 4,
                                 double, 2,
                                 vreinterpretq_f64_f32)
#endif

}

#endif
