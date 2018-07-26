/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_NEON_INT32_HPP
#define XSIMD_NEON_INT32_HPP

#include <utility>

#include "xsimd_base.hpp"
#include "xsimd_neon_bool.hpp"

namespace xsimd
{
    /*********************
     * batch<int32_t, 4> *
     *********************/

    template <>
    struct simd_batch_traits<batch<int32_t, 4>>
    {
        using value_type = int32_t;
        static constexpr std::size_t size = 4;
        using batch_bool_type = batch_bool<int32_t, 4>;
        static constexpr std::size_t align = XSIMD_DEFAULT_ALIGNMENT;
    };

    template <>
    class batch<int32_t, 4> : public simd_batch<batch<int32_t, 4>>
    {
        using simd_type = int32x4_t;

    public:

        using self_type = batch<int32_t, 4>;
        using base_type = simd_batch<self_type>;

        batch();
        explicit batch(int32_t d);
        batch(int32_t d0, int32_t d1, int32_t d2, int32_t d3);
        explicit batch(const int32_t* src);

        batch(const int32_t* src, aligned_mode);
        batch(const int32_t* src, unaligned_mode);

        batch(const simd_type& rhs);
        batch& operator=(const simd_type& rhs);

        operator simd_type() const;

        batch& load_aligned(const float* src);
        batch& load_unaligned(const float* src);

        batch& load_aligned(const double* src);
        batch& load_unaligned(const double* src);

        batch& load_aligned(const int32_t* src);
        batch& load_unaligned(const int32_t* src);

        batch& load_aligned(const int64_t* src);
        batch& load_unaligned(const int64_t* src);

        batch& load_aligned(const int8_t* src);
        batch& load_unaligned(const int8_t* src);

        batch& load_aligned(const uint8_t* src);
        batch& load_unaligned(const uint8_t* src);

        void store_aligned(float* dst) const;
        void store_unaligned(float* dst) const;

        void store_aligned(double* dst) const;
        void store_unaligned(double* dst) const;

        void store_aligned(int32_t* dst) const;
        void store_unaligned(int32_t* dst) const;

        void store_aligned(int64_t* dst) const;
        void store_unaligned(int64_t* dst) const;

        void store_aligned(int8_t* dst) const;
        void store_unaligned(int8_t* dst) const;

        void store_aligned(uint8_t* dst) const;
        void store_unaligned(uint8_t* dst) const;

        using base_type::load_aligned;
        using base_type::load_unaligned;
        using base_type::store_aligned;
        using base_type::store_unaligned;

        int32_t operator[](std::size_t index) const;

    private:

        simd_type m_value;
    };

    batch<int32_t, 4> operator<<(const batch<int32_t, 4>& lhs, int32_t rhs);
    batch<int32_t, 4> operator>>(const batch<int32_t, 4>& lhs, int32_t rhs);

    /************************************
     * batch<int32_t, 4> implementation *
     ************************************/

    inline batch<int32_t, 4>::batch()
    {
    }

    inline batch<int32_t, 4>::batch(int32_t d)
        : m_value(vdupq_n_s32(d))
    {
    }

    inline batch<int32_t, 4>::batch(int32_t d1, int32_t d2, int32_t d3, int32_t d4)
        : m_value{d1, d2, d3, d4}
    {
    }

    inline batch<int32_t, 4>::batch(const int32_t* d)
        : m_value(vld1q_s32(d))
    {
    }

    inline batch<int32_t, 4>::batch(const int32_t* d, aligned_mode)
        : batch(d)
    {
    }

    inline batch<int32_t, 4>::batch(const int32_t* d, unaligned_mode)
        : batch(d)
    {
    }

    inline batch<int32_t, 4>::batch(const simd_type& rhs)
        : m_value(rhs)
    {
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::operator=(const simd_type& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_aligned(const float* d)
    {
        m_value = vcvtq_s32_f32(vld1q_f32(d));
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_unaligned(const float* d)
    {
        m_value = vcvtq_s32_f32(vld1q_f32(d));
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_aligned(const double* d)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        float32x2_t tmp_l = vcvtx_f32_f64(float64x2_t{d[0], d[1]});
        float32x2_t tmp_h = vcvtx_f32_f64(float64x2_t{d[2], d[3]});
        m_value = vcvtq_s32_f32(vcombine_f32(tmp_l, tmp_h));
        return *this;
    #else
        m_value = int32x4_t{
            static_cast<int32_t>(d[0]),
            static_cast<int32_t>(d[1]),
            static_cast<int32_t>(d[2]),
            static_cast<int32_t>(d[3])
        };
    #endif
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_unaligned(const double* d)
    {
        return load_aligned(d);
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_aligned(const int32_t* d)
    {
        m_value = vld1q_s32(d);
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_unaligned(const int32_t* d)
    {
        m_value = vld1q_s32(d);
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_aligned(const int64_t* d)
    {
        m_value = int32x4_t{
            static_cast<int32_t>(d[0]),
            static_cast<int32_t>(d[1]),
            static_cast<int32_t>(d[2]),
            static_cast<int32_t>(d[3])
        };
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_unaligned(const int64_t* d)
    {
        load_aligned(d);
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_aligned(const int8_t* src)
    {
        int8x8_t tmp = vld1_s8((const int8_t*)src);
        int16x8_t tmp2 = vmovl_s8(tmp);
        int16x4_t tmp3 = vget_low_s16(tmp2);
        m_value = vmovl_s16(tmp3);
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_unaligned(const int8_t* src)
    {
        return load_aligned(src);
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_aligned(const uint8_t* src)
    {
        uint8x8_t tmp = vld1_u8((const uint8_t*)src);
        uint16x8_t tmp2 = vmovl_u8(tmp);
        uint16x4_t tmp3 = vget_low_u16(tmp2);
        m_value = vmovl_u16(tmp3);
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_unaligned(const uint8_t* src)
    {
        return load_aligned(src);
    }

    inline void batch<int32_t, 4>::store_aligned(double* dst) const
    {
        alignas(16) int32_t tmp[4];
        vst1q_s32(tmp, m_value);
        dst[0] = static_cast<double>(tmp[0]);
        dst[1] = static_cast<double>(tmp[1]);
        dst[2] = static_cast<double>(tmp[2]);
        dst[3] = static_cast<double>(tmp[3]);
    }

    inline void batch<int32_t, 4>::store_unaligned(double* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int32_t, 4>::store_aligned(int32_t* dst) const
    {
        vst1q_s32(dst, m_value);
    }

    inline void batch<int32_t, 4>::store_unaligned(int32_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int32_t, 4>::store_aligned(int64_t* dst) const
    {
        alignas(16) int32_t tmp[4];
        vst1q_s32(tmp, m_value);
        dst[0] = static_cast<int64_t>(tmp[0]);
        dst[1] = static_cast<int64_t>(tmp[1]);
        dst[2] = static_cast<int64_t>(tmp[2]);
        dst[3] = static_cast<int64_t>(tmp[3]);
    }

    inline void batch<int32_t, 4>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int32_t, 4>::store_aligned(int8_t* dst) const
    {
        int16x4_t tmp2 = vmovn_s32(m_value);
        int16x8_t tmp3 = vcombine_s16(tmp2, vdup_n_s16(0));
        int8x8_t tmp4 = vmovn_s16(tmp3);
        vst1_s8((int8_t*)dst, tmp4);
    }

    inline void batch<int32_t, 4>::store_unaligned(int8_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int32_t, 4>::store_aligned(uint8_t* dst) const
    {
        uint16x4_t tmp2 = vmovn_u32(m_value);
        uint16x8_t tmp3 = vcombine_u16(tmp2, vdup_n_u16(0));
        uint8x8_t tmp4 = vmovn_u16(tmp3);
        vst1_u8((uint8_t*)dst, tmp4);
    }

    inline void batch<int32_t, 4>::store_unaligned(uint8_t* dst) const
    {
        store_aligned(dst);
    }

    inline batch<int32_t, 4>::operator int32x4_t() const
    {
        return m_value;
    }

    inline int32_t batch<int32_t, 4>::operator[](std::size_t index) const
    {
        return m_value[index];
    }



    namespace detail
    {
        template <>
        struct batch_kernel<int32_t, 4>
        {
            using batch_type = batch<int32_t, 4>;
            using value_type = int32_t;
            using batch_bool_type = batch_bool<int32_t, 4>;

            static batch_type neg(const batch_type& rhs)
            {
                return vnegq_s32(rhs);
            }

            static batch_type add(const batch_type& lhs, const batch_type& rhs)
            {
                return vaddq_s32(lhs, rhs);
            }

            static batch_type sub(const batch_type& lhs, const batch_type& rhs)
            {
                return vsubq_s32(lhs, rhs);
            }

            static batch_type mul(const batch_type& lhs, const batch_type& rhs)
            {
                return vmulq_s32(lhs, rhs);
            }

            static batch_type div(const batch_type& lhs, const batch_type& rhs)
            {
#if defined(XSIMD_FAST_INTEGER_DIVISION)
                return vcvtq_s32_f32(vcvtq_f32_s32(lhs) / vcvtq_f32_s32(rhs));
#else
                return int32x4_t{
                    lhs[0] / rhs[0], lhs[1] / rhs[1],
                    lhs[2] / rhs[2], lhs[3] / rhs[3]
                };
#endif
            }

            static batch_type mod(const batch_type& lhs, const batch_type& rhs)
            {
                return int32x4_t{
                    lhs[0] % rhs[0], lhs[1] % rhs[1],
                    lhs[2] % rhs[2], lhs[3] % rhs[3]
                };
            }

            static batch_bool_type eq(const batch_type& lhs, const batch_type& rhs)
            {
                return vceqq_s32(lhs, rhs);
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
                return !(lhs == rhs);
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
                return vcltq_s32(lhs, rhs);
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
                return vcleq_s32(lhs, rhs);
            }

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return vandq_s32(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return vorrq_s32(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return veorq_s32(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return vmvnq_s32(rhs);
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return vbicq_s32(lhs, rhs);
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
                return vminq_s32(lhs, rhs);
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
                return vmaxq_s32(lhs, rhs);
            }

            static batch_type abs(const batch_type& rhs)
            {
                return vabsq_s32(rhs);
            }

            static batch_type fma(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return x * y + z;
            }

            static batch_type fms(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return x * y - z;
            }

            static batch_type fnma(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return -x * y + z;
            }

            static batch_type fnms(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return -x * y - z;
            }

            static value_type hadd(const batch_type& rhs)
            {
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
                return vaddvq_s32(rhs);
#else
                int32x2_t tmp = vpadd_s32(vget_low_s32(rhs), vget_high_s32(rhs));
                tmp = vpadd_s32(tmp, tmp);
                return vget_lane_s32(tmp, 0);
#endif
            }

            static batch_type select(const batch_bool_type& cond, const batch_type& a, const batch_type& b)
            {
                return vbslq_s32(cond, a, b);
            }
        };
    }


    /*inline batch<int32_t, 4> haddp(const batch<int32_t, 4>* row)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        int32x4_t tmp1 = vpaddq_s32(row[0], row[1]);
        int32x4_t tmp2 = vpaddq_s32(row[2], row[3]);
        return vpaddq_s32(tmp1, tmp2);
    #else
        // row = (a,b,c,d)
        int32x2_t tmp1, tmp2, tmp3;
        // tmp1 = (a0 + a2, a1 + a3)
        tmp1 = vpadd_s32(vget_low_s32(row[0]), vget_high_s32(row[0]));
        // tmp2 = (b0 + b2, b1 + b3)
        tmp2 = vpadd_s32(vget_low_s32(row[1]), vget_high_s32(row[1]));
        // tmp1 = (a0..3, b0..3)
        tmp1 = vpadd_s32(tmp1, tmp2);
        // tmp2 = (c0 + c2, c1 + c3)
        tmp2 = vpadd_s32(vget_low_s32(row[2]), vget_high_s32(row[2]));
        // tmp3 = (d0 + d2, d1 + d3)
        tmp3 = vpadd_s32(vget_low_s32(row[3]), vget_high_s32(row[3]));
        // tmp1 = (c0..3, d0..3)
        tmp2 = vpadd_s32(tmp2, tmp3);
        // return = (a0..3, b0..3, c0..3, d0..3)
        return vcombine_s32(tmp1, tmp2);
    #endif
    }*/

    namespace detail
    {
        inline batch<int32_t, 4> shift_left(const batch<int32_t, 4>& lhs, const int n)
        {
            switch(n)
            {
                case 0: return lhs;
                REPEAT_32(vshlq_n_s32, 0);
                default: break;
            }
            return batch<int32_t, 4>(0);
        }

        inline batch<int32_t, 4> shift_right(const batch<int32_t, 4>& lhs, const int n)
        {
            switch(n)
            {
                case 0: return lhs;
                REPEAT_32(vshrq_n_s32, 0);
                default: break;
            }
            return batch<int32_t, 4>(0);
        }
    }

    inline batch<int32_t, 4> operator<<(const batch<int32_t, 4>& lhs, int32_t rhs)
    {
        return detail::shift_left(lhs, rhs);
    }

    inline batch<int32_t, 4> operator>>(const batch<int32_t, 4>& lhs, int32_t rhs)
    {
        return detail::shift_right(lhs, rhs);
    }

    inline batch<int32_t, 4> operator<<(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vshlq_s32(lhs, rhs);
    }

}

#endif