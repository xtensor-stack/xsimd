/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_NEON_INT64_HPP
#define XSIMD_NEON_INT64_HPP

#include "xsimd_base.hpp"

namespace xsimd
{
    template <>
    struct simd_batch_traits<batch<int64_t, 2>>
    {
        using value_type = int64_t;
        static constexpr std::size_t size = 2;
        using batch_bool_type = batch_bool<int64_t, 2>;
    };

    template <>
    class batch<int64_t, 2> : public simd_batch<batch<int64_t, 2>>
    {
        using simd_type = int64x2_t;

    public:

        batch();
        explicit batch(int64_t d);
        batch(int64_t d0, int64_t d1);
        explicit batch(const int64_t* src);

        batch(const int64_t* src, aligned_mode);
        batch(const int64_t* src, unaligned_mode);

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

        int64_t operator[](std::size_t index) const;

    private:

        simd_type m_value;
    };

    /**
     * Implementation of batch
     */

    inline batch<int64_t, 2>::batch()
    {
    }

    inline batch<int64_t, 2>::batch(int64_t d)
        : m_value(vdupq_n_s64(d))
    {
    }

    inline batch<int64_t, 2>::batch(int64_t d1, int64_t d2)
        : m_value{d1, d2}
    {
    }

    inline batch<int64_t, 2>::batch(const int64_t* d)
        : m_value(vld1q_s64(d))
    {
    }

    inline batch<int64_t, 2>::batch(const int64_t* d, aligned_mode)
        : batch(d)
    {
    }

    inline batch<int64_t, 2>::batch(const int64_t* d, unaligned_mode)
        : batch(d)
    {
    }

    inline batch<int64_t, 2>::batch(const simd_type& rhs)
        : m_value(rhs)
    {
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::operator=(const simd_type& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_aligned(const float* d)
    {
        m_value = vcvtq_s64_f64(vcvt_f64_f32(vld1_f32(d)));
        return *this;
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_unaligned(const float* d)
    {
        return load_unaligned(d);
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_aligned(const double* d)
    {
        m_value = vcvtq_s64_f64(vld1q_f64(d));
        return *this;
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_unaligned(const double* d)
    {
        return load_aligned(d);
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_aligned(const int32_t* d)
    {
        m_value = int64x2_t{d[0], d[1]};
        return *this;
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_unaligned(const int32_t* d)
    {
        return load_aligned(d);
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_aligned(const int64_t* d)
    {
        m_value = vld1q_s64(d);
        return *this;
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_unaligned(const int64_t* d)
    {
        return load_aligned(d);
    }

    inline void batch<int64_t, 2>::store_aligned(float* dst) const
    {
    #ifdef XSIMD_ARM_64
        vst1_f32(dst, vcvt_f32_f64(vcvtq_f64_s64(m_value)));
    #else
        dst[0] = static_cast<float>(m_value[0]);
        dst[1] = static_cast<float>(m_value[1]);
    #endif
    }

    inline void batch<int64_t, 2>::store_unaligned(float* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int64_t, 2>::store_aligned(double* dst) const
    {
        vst1q_f64(dst, vcvtq_f64_s64(m_value));
    }

    inline void batch<int64_t, 2>::store_unaligned(double* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int64_t, 2>::store_aligned(int32_t* dst) const
    {
        dst[0] = static_cast<int32_t>(m_value[0]);
        dst[1] = static_cast<int32_t>(m_value[1]);
    }

    inline void batch<int64_t, 2>::store_unaligned(int32_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int64_t, 2>::store_aligned(int64_t* dst) const
    {
        vst1q_s64(dst, m_value);
    }

    inline void batch<int64_t, 2>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline batch<int64_t, 2>::operator int64x2_t() const
    {
        return m_value;
    }

    inline int64_t batch<int64_t, 2>::operator[](std::size_t index) const
    {
        return m_value[index];
    }

    inline batch<int64_t, 2> operator-(const batch<int64_t, 2>& lhs)
    {
        return vnegq_s64(lhs);
    }

    inline batch<int64_t, 2> operator+(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return vaddq_s64(lhs, rhs);
    }

    inline batch<int64_t, 2> operator-(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return vsubq_s64(lhs, rhs);
    }

    inline batch<int64_t, 2> operator*(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return {lhs[0] * rhs[0], lhs[1] * rhs[1]};
    }

    inline batch<int64_t, 2> operator/(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return vcvtq_s64_f64(vcvtq_f64_s64(lhs) / vcvtq_f64_s64(rhs));
    }

    inline batch<int64_t, 2> min(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return {lhs[0] < rhs[0] ? lhs[0] : rhs[0], 
                lhs[1] < rhs[1] ? lhs[1] : rhs[1]};
    }

    inline batch<int64_t, 2> max(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return {lhs[0] > rhs[0] ? lhs[0] : rhs[0], 
                lhs[1] > rhs[1] ? lhs[1] : rhs[1]};
    }

    inline batch<int64_t, 2> abs(const batch<int64_t, 2>& lhs)
    {
        return vabsq_s64(lhs);
    }

    inline batch<int64_t, 2> fma(const batch<int64_t, 2>& x, const batch<int64_t, 2>& y, const batch<int64_t, 2>& z)
    {
        return x * y + z;
    }

    inline batch<int64_t, 2> fms(const batch<int64_t, 2>& x, const batch<int64_t, 2>& y, const batch<int64_t, 2>& z)
    {
        return x * y - z;
    }

    inline batch<int64_t, 2> fnma(const batch<int64_t, 2>& x, const batch<int64_t, 2>& y, const batch<int64_t, 2>& z)
    {
        return -x * y + z;
    }

    inline batch<int64_t, 2> fnms(const batch<int64_t, 2>& x, const batch<int64_t, 2>& y, const batch<int64_t, 2>& z)
    {
        return -x * y - z;
    }

    inline int64_t hadd(const batch<int64_t, 2>& rhs)
    {
    #ifdef XSIMD_ARM_64
        return vaddvq_s64(rhs);
    #else
        int64x2_t tmp = vpaddq_s64(rhs, rhs);
        return vget_lane_s64(tmp, 0);
    #endif
    }

    inline batch<int64_t, 2> haddp(const batch<int64_t, 2>* row)
    {
        return vpaddq_s64(row[0], row[1]);
    }

    inline batch<int64_t, 2> operator<<(const batch<int64_t, 2>& lhs, int64_t rhs)
    {
        return vshlq_n_s64(lhs, rhs);
    }

    inline batch<int64_t, 2> operator>>(const batch<int64_t, 2>& lhs, int64_t rhs)
    {
        return vshrq_n_s64(lhs, rhs);
    }

    inline batch_bool<int64_t, 2> isnan(const batch<int64_t, 2>& x)
    {
        return !vceqq_s64(x, x);
    }

    inline batch_bool<int64_t, 2> operator==(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return vceqq_s64(lhs, rhs);
    }

    inline batch_bool<int64_t, 2> operator!=(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return !(lhs == rhs);
    }

    inline batch_bool<int64_t, 2> operator<(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return vcltq_s64(lhs, rhs);
    }

    inline batch_bool<int64_t, 2> operator<=(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return vcleq_s64(lhs, rhs);
    }

    inline batch_bool<int64_t, 2> operator>(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return vcgtq_s64(lhs, rhs);
    }

    inline batch_bool<int64_t, 2> operator>=(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return vcgeq_s64(lhs, rhs);
    }

    inline batch<int64_t, 2> select(const batch_bool<int64_t, 2>& cond, const batch<int64_t, 2>& a, const batch<int64_t, 2>& b)
    {
        return vbslq_s64(cond, a, b);
    }

    inline batch<int64_t, 2> operator&(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return vandq_s64(lhs, rhs);
    }

    inline batch<int64_t, 2> operator|(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return vorrq_s64(lhs, rhs);
    }

    inline batch<int64_t, 2> operator^(const batch<int64_t, 2>& lhs, const batch<int64_t, 2>& rhs)
    {
        return veorq_s64(lhs, rhs);
    }

    inline batch<int64_t, 2> operator~(const batch<int64_t, 2>& rhs)
    {
        return vreinterpretq_s64_s32(vmvnq_s32(vreinterpretq_s32_s64(rhs)));
    }
}

#endif
