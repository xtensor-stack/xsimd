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
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    #include "xsimd_neon_double.hpp"
#endif

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

    /**************************
     * bitwise cast functions *
     **************************/

    template <class B>
    B bitwise_cast(const batch<float, 4>& x);

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    template <class B>
    B bitwise_cast(const batch<double, 2>& x);
#endif

    template <class B>
    B bitwise_cast(const batch<int32_t, 4>& x);

    template <class B>
    B bitwise_cast(const batch<int64_t, 2>& x);

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

    template <>
    inline batch<int32_t, 4> bitwise_cast(const batch<float, 4>& x)
    {
        return vreinterpretq_s32_f32(x);
    }

    template <>
    inline batch<int64_t, 2> bitwise_cast(const batch<float, 4>& x)
    {
        return vreinterpretq_s64_f32(x);
    }

    template <>
    inline batch<float, 4> bitwise_cast(const batch<int32_t, 4>& x)
    {
        return vreinterpretq_f32_s32(x);
    }


    template <>
    inline batch<float, 4> bitwise_cast(const batch<int64_t, 2>& x)
    {
        return vreinterpretq_f32_s64(x);
    }

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    template <>
    inline batch<float, 4> bitwise_cast(const batch<double, 2>& x)
    {
        return vreinterpretq_f32_f64(x);
    }

    template <>
    inline batch<int32_t, 4> bitwise_cast(const batch<double, 2>& x)
    {
        return vreinterpretq_s32_f64(x);
    }

    template <>
    inline batch<int64_t, 2> bitwise_cast(const batch<double, 2>& x)
    {
        return vreinterpretq_s64_f64(x);
    }

    template <>
    inline batch<double, 2> bitwise_cast(const batch<int32_t, 4>& x)
    {
        return vreinterpretq_f64_s32(x);
    }

    template <>
    inline batch<double, 2> bitwise_cast(const batch<int64_t, 2>& x)
    {
        return vreinterpretq_f64_s64(x);
    }

    template <>
    inline batch<double, 2> bitwise_cast(const batch<float, 4>& x)
    {
        return vreinterpretq_f64_f32(x);
    }
#endif

}

#endif
