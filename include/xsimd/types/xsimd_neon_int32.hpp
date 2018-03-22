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

        void store_aligned(float* dst) const;
        void store_unaligned(float* dst) const;

        void store_aligned(double* dst) const;
        void store_unaligned(double* dst) const;

        void store_aligned(int32_t* dst) const;
        void store_unaligned(int32_t* dst) const;

        void store_aligned(int64_t* dst) const;
        void store_unaligned(int64_t* dst) const;

        int32_t operator[](std::size_t index) const;

    private:

        simd_type m_value;
    };

    /**
     * Implementation of batch
     */

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

    inline batch<int32_t, 4>::operator int32x4_t() const
    {
        return m_value;
    }

    inline int32_t batch<int32_t, 4>::operator[](std::size_t index) const
    {
        return m_value[index];
    }

    inline batch<int32_t, 4> operator-(const batch<int32_t, 4>& lhs)
    {
        return vnegq_s32(lhs);
    }

    inline batch<int32_t, 4> operator+(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vaddq_s32(lhs, rhs);
    }

    inline batch<int32_t, 4> operator-(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vsubq_s32(lhs, rhs);
    }

    inline batch<int32_t, 4> operator*(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vmulq_s32(lhs, rhs);
    }

    inline batch<int32_t, 4> operator/(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vcvtq_s32_f32(vcvtq_f32_s32(lhs) / vcvtq_f32_s32(rhs));
    }

    inline batch<int32_t, 4> min(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vminq_s32(lhs, rhs);
    }

    inline batch<int32_t, 4> max(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vmaxq_s32(lhs, rhs);
    }

    inline batch<int32_t, 4> fmin(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return min(lhs, rhs);
    }

    inline batch<int32_t, 4> fmax(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return max(lhs, rhs);
    }

    inline batch<int32_t, 4> abs(const batch<int32_t, 4>& lhs)
    {
        return vabsq_s32(lhs);
    }

    inline batch<int32_t, 4> fma(const batch<int32_t, 4>& x, const batch<int32_t, 4>& y, const batch<int32_t, 4>& z)
    {
        return x * y + z;
    }

    inline batch<int32_t, 4> fms(const batch<int32_t, 4>& x, const batch<int32_t, 4>& y, const batch<int32_t, 4>& z)
    {
        return x * y - z;
    }

    inline batch<int32_t, 4> fnma(const batch<int32_t, 4>& x, const batch<int32_t, 4>& y, const batch<int32_t, 4>& z)
    {
        return -x * y + z;
    }

    inline batch<int32_t, 4> fnms(const batch<int32_t, 4>& x, const batch<int32_t, 4>& y, const batch<int32_t, 4>& z)
    {
        return -x * y - z;
    }

    inline int32_t hadd(const batch<int32_t, 4>& rhs)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        return vaddvq_s32(rhs);
    #else
        int32x2_t tmp = vpadd_s32(vget_low_s32(rhs), vget_high_s32(rhs));
        tmp = vpadd_s32(tmp, tmp);
        return vget_lane_s32(tmp, 0);
    #endif
    }

    inline batch<int32_t, 4> haddp(const batch<int32_t, 4>* row)
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
    }

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

    inline batch_bool<int32_t, 4> operator==(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vceqq_s32(lhs, rhs);
    }

    inline batch_bool<int32_t, 4> operator!=(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return !(lhs == rhs);
    }

    inline batch_bool<int32_t, 4> operator<(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vcltq_s32(lhs, rhs);
    }

    inline batch_bool<int32_t, 4> operator<=(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vcleq_s32(lhs, rhs);
    }

    inline batch_bool<int32_t, 4> operator>(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vcgtq_s32(lhs, rhs);
    }

    inline batch_bool<int32_t, 4> operator>=(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vcgeq_s32(lhs, rhs);
    }

    inline batch<int32_t, 4> select(const batch_bool<int32_t, 4>& cond, const batch<int32_t, 4>& a, const batch<int32_t, 4>& b)
    {
        return vbslq_s32(cond, a, b);
    }

    inline batch<int32_t, 4> operator&(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vandq_s32(lhs, rhs);
    }

    inline batch<int32_t, 4> operator|(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return vorrq_s32(lhs, rhs);
    }

    inline batch<int32_t, 4> operator^(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return veorq_s32(lhs, rhs);
    }

    inline batch<int32_t, 4> operator~(const batch<int32_t, 4>& rhs)
    {
        return vmvnq_s32(rhs);
    }
}

#endif