/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_NEON_FLOAT_HPP
#define XSIMD_NEON_FLOAT_HPP

#include "xsimd_base.hpp"
#include "xsimd_neon_bool.hpp"

namespace xsimd
{
    template <>
    struct simd_batch_traits<batch<float, 4>>
    {
        using value_type = float;
        static constexpr std::size_t size = 4;
        using batch_bool_type = batch_bool<float, 4>;
        static constexpr std::size_t align = XSIMD_DEFAULT_ALIGNMENT;
    };

    template <>
    class batch<float, 4> : public simd_batch<batch<float, 4>>
    {
        using simd_type = float32x4_t;
    public:

        batch();
        explicit batch(float d);
        batch(float d0, float d1, float d2, float d3);
        explicit batch(const float* src);

        batch(const float* src, aligned_mode);
        batch(const float* src, unaligned_mode);

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

        float operator[](std::size_t index) const;

    private:

        simd_type m_value;
    };

    batch<float, 4> operator-(const batch<float, 4>& rhs);
    batch<float, 4> operator+(const batch<float, 4>& lhs, const batch<float, 4>& rhs);
    batch<float, 4> operator-(const batch<float, 4>& lhs, const batch<float, 4>& rhs);
    batch<float, 4> operator*(const batch<float, 4>& lhs, const batch<float, 4>& rhs);
    batch<float, 4> operator/(const batch<float, 4>& lhs, const batch<float, 4>& rhs);

    batch_bool<float, 4> operator==(const batch<float, 4>& lhs, const batch<float, 4>& rhs);
    batch_bool<float, 4> operator!=(const batch<float, 4>& lhs, const batch<float, 4>& rhs);
    batch_bool<float, 4> operator<(const batch<float, 4>& lhs, const batch<float, 4>& rhs);
    batch_bool<float, 4> operator<=(const batch<float, 4>& lhs, const batch<float, 4>& rhs);

    batch<float, 4> operator&(const batch<float, 4>& lhs, const batch<float, 4>& rhs);
    batch<float, 4> operator|(const batch<float, 4>& lhs, const batch<float, 4>& rhs);
    batch<float, 4> operator^(const batch<float, 4>& lhs, const batch<float, 4>& rhs);
    batch<float, 4> operator~(const batch<float, 4>& rhs);
    batch<float, 4> bitwise_andnot(const batch<float, 4>& lhs, const batch<float, 4>& rhs);

    batch<float, 4> min(const batch<float, 4>& lhs, const batch<float, 4>& rhs);
    batch<float, 4> max(const batch<float, 4>& lhs, const batch<float, 4>& rhs);
    batch<float, 4> fmin(const batch<float, 4>& lhs, const batch<float, 4>& rhs);
    batch<float, 4> fmax(const batch<float, 4>& lhs, const batch<float, 4>& rhs);

    batch<float, 4> abs(const batch<float, 4>& rhs);
    batch<float, 4> fabs(const batch<float, 4>& rhs);
    batch<float, 4> sqrt(const batch<float, 4>& rhs);

    batch<float, 4> fma(const batch<float, 4>& x, const batch<float, 4>& y, const batch<float, 4>& z);
    batch<float, 4> fms(const batch<float, 4>& x, const batch<float, 4>& y, const batch<float, 4>& z);
    batch<float, 4> fnma(const batch<float, 4>& x, const batch<float, 4>& y, const batch<float, 4>& z);
    batch<float, 4> fnms(const batch<float, 4>& x, const batch<float, 4>& y, const batch<float, 4>& z);

    float hadd(const batch<float, 4>& rhs);
    batch<float, 4> haddp(const batch<float, 4>* row);

    batch<float, 4> select(const batch_bool<float, 4>& cond, const batch<float, 4>& a, const batch<float, 4>& b);

    batch_bool<float, 4> isnan(const batch<float, 4>& x);

    /**
     * Implementation of batch
     */

    inline batch<float, 4>::batch()
    {
    }

    inline batch<float, 4>::batch(float d)
        : m_value(vdupq_n_f32(d))
    {
    }

    inline batch<float, 4>::batch(float d1, float d2, float d3, float d4)
        : m_value{d1, d2, d3, d4}
    {
    }

    inline batch<float, 4>::batch(const float* d)
        : m_value(vld1q_f32(d))
    {
    }

    inline batch<float, 4>::batch(const float* d, aligned_mode)
        : batch(d)
    {
    }

    inline batch<float, 4>::batch(const float* d, unaligned_mode)
        : batch(d)
    {
    }

    inline batch<float, 4>::batch(const simd_type& rhs)
        : m_value(rhs)
    {
    }

    inline batch<float, 4>& batch<float, 4>::operator=(const simd_type& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline void batch<float, 4>::store_aligned(float* dst) const
    {
        vst1q_f32(dst, m_value);
    }

    inline void batch<float, 4>::store_unaligned(float* dst) const
    {
        store_aligned(dst);
    }

    inline batch<float, 4>& batch<float, 4>::load_aligned(const float* d)
    {
        m_value = vld1q_f32(d);
        return *this;
    }

    inline batch<float, 4>& batch<float, 4>::load_unaligned(const float* d)
    {
        return load_aligned(d);
    }

    inline batch<float, 4>& batch<float, 4>::load_aligned(const double* d)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        float32x2_t tmp_l = vcvt_f32_f64(vld1q_f64(&d[0]));
        float32x2_t tmp_h = vcvt_f32_f64(vld1q_f64(&d[2]));
        m_value = vcombine_f32(tmp_l, tmp_h);
        return *this;
    #else
        m_value = float32x4_t{
            static_cast<float>(d[0]),
            static_cast<float>(d[1]),
            static_cast<float>(d[2]),
            static_cast<float>(d[3])
        };
    #endif
        return *this;
    }

    inline batch<float, 4>& batch<float, 4>::load_unaligned(const double* d)
    {
        return load_aligned(d);
    }

    inline batch<float, 4>& batch<float, 4>::load_aligned(const int32_t* d)
    {
        m_value = vcvtq_f32_s32(vld1q_s32(d));
        return *this;
    }

    inline batch<float, 4>& batch<float, 4>::load_unaligned(const int32_t* d)
    {
        return load_aligned(d);
    }

    inline batch<float, 4>& batch<float, 4>::load_aligned(const int64_t* d)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        float32x2_t tmp_l = vcvt_f32_f64(vcvtq_f64_s64(vld1q_s64(&d[0])));
        float32x2_t tmp_h = vcvt_f32_f64(vcvtq_f64_s64(vld1q_s64(&d[2])));
        m_value = vcombine_f32(tmp_l, tmp_h);
    #else
        m_value = float32x4_t{
            static_cast<float>(d[0]),
            static_cast<float>(d[1]),
            static_cast<float>(d[2]),
            static_cast<float>(d[3])
        };
    #endif
        return *this;
    }

    inline batch<float, 4>& batch<float, 4>::load_unaligned(const int64_t* d)
    {
        return load_aligned(d);
    }

    inline void batch<float, 4>::store_aligned(double* dst) const
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        float64x2_t tmp_l = vcvt_f64_f32(vget_low_f32(m_value));
        float64x2_t tmp_h = vcvt_f64_f32(vget_high_f32(m_value));
        vst1q_f64(&(dst[0]), tmp_l);
        vst1q_f64(&(dst[2]), tmp_h);
    #else
        dst[0] = static_cast<double>(m_value[0]);
        dst[1] = static_cast<double>(m_value[1]);
        dst[2] = static_cast<double>(m_value[2]);
        dst[3] = static_cast<double>(m_value[3]);
    #endif
    }

    inline void batch<float, 4>::store_unaligned(double* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<float, 4>::store_aligned(int32_t* dst) const
    {
        vst1q_s32(dst, vcvtq_s32_f32(m_value));
    }

    inline void batch<float, 4>::store_unaligned(int32_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<float, 4>::store_aligned(int64_t* dst) const
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        int64x2_t tmp_l = vcvtq_s64_f64(vcvt_f64_f32(vget_low_f32(m_value)));
        int64x2_t tmp_h = vcvtq_s64_f64(vcvt_f64_f32(vget_high_f32(m_value)));
        vst1q_s64(&(dst[0]), tmp_l);
        vst1q_s64(&(dst[2]), tmp_h);
    #else
        dst[0] = static_cast<double>(m_value[0]);
        dst[1] = static_cast<double>(m_value[1]);
        dst[2] = static_cast<double>(m_value[2]);
        dst[3] = static_cast<double>(m_value[3]);
    #endif
    }

    inline void batch<float, 4>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline batch<float, 4>::operator float32x4_t() const
    {
        return m_value;
    }

    inline float batch<float, 4>::operator[](std::size_t index) const
    {
        return m_value[index];
    }

    inline batch<float, 4> operator-(const batch<float, 4>& lhs)
    {
        return vnegq_f32(lhs);
    }

    inline batch<float, 4> operator+(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vaddq_f32(lhs, rhs);
    }

    inline batch<float, 4> operator-(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vsubq_f32(lhs, rhs);
    }

    inline batch<float, 4> operator*(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vmulq_f32(lhs, rhs);
    }

    inline batch<float, 4> operator/(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        return vdivq_f32(lhs, rhs);
    #else
        // from stackoverflow & https://projectne10.github.io/Ne10/doc/NE10__divc_8neon_8c_source.html
        // get an initial estimate of 1/b.
        float32x4_t reciprocal = vrecpeq_f32(rhs);

        // use a couple Newton-Raphson steps to refine the estimate.  Depending on your
        // application's accuracy requirements, you may be able to get away with only
        // one refinement (instead of the two used here).  Be sure to test!
        reciprocal = vmulq_f32(vrecpsq_f32(rhs, reciprocal), reciprocal);
        reciprocal = vmulq_f32(vrecpsq_f32(rhs, reciprocal), reciprocal);

        // and finally, compute a / b = a * (1 / b)
        return vmulq_f32(lhs, reciprocal);
    #endif
    }

    inline batch<float, 4> min(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vminq_f32(lhs, rhs);
    }

    inline batch<float, 4> max(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vmaxq_f32(lhs, rhs);
    }

    inline batch<float, 4> fmin(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return min(lhs, rhs);
    }

    inline batch<float, 4> fmax(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return max(lhs, rhs);
    }

    inline batch<float, 4> abs(const batch<float, 4>& lhs)
    {
        return vabsq_f32(lhs);
    }

    inline batch<float, 4> fabs(const batch<float, 4>& lhs)
    {
        return abs(lhs);
    }

    inline batch<float, 4> sqrt(const batch<float, 4>& lhs)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        return vsqrtq_f32(lhs);
    #else
        batch<float, 4> sqrt_reciprocal = vrsqrteq_f32(lhs);
        // one iter
        // sqrt_reciprocal = sqrt_reciprocal * vrsqrtsq_f32(lhs * sqrt_reciprocal, sqrt_reciprocal);
        batch<float, 4> sqrt_approx = lhs * sqrt_reciprocal * batch<float, 4>(vrsqrtsq_f32(lhs * sqrt_reciprocal, sqrt_reciprocal));
        batch<float, 4> zero(0.f);
        return select(lhs == zero, zero, sqrt_approx);
    #endif
    }

    inline batch<float, 4> fma(const batch<float, 4>& x, const batch<float, 4>& y, const batch<float, 4>& z)
    {
    #ifdef __ARM_FEATURE_FMA
        // TODO check if destructive!
        // multiplies x * y and accumulates into z 
        return vfmaq_f32(z, x, y);
    #else
        return x * y + z;
    #endif
    }

    inline batch<float, 4> fms(const batch<float, 4>& x, const batch<float, 4>& y, const batch<float, 4>& z)
    {
    #ifdef __ARM_FEATURE_FMA
        return vfmaq_f32(-z, x, y);
    #else
        return x * y - z;
    #endif
    }

    inline batch<float, 4> fnma(const batch<float, 4>& x, const batch<float, 4>& y, const batch<float, 4>& z)
    {
        return fma(-x, y, z);
    }

    inline batch<float, 4> fnms(const batch<float, 4>& x, const batch<float, 4>& y, const batch<float, 4>& z)
    {
        return fms(-x, y, z);
    }

    inline float hadd(const batch<float, 4>& rhs)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        return vaddvq_f32(rhs);
    #else
        float32x2_t tmp = vpadd_f32(vget_low_f32(rhs), vget_high_f32(rhs));
        tmp = vpadd_f32(tmp, tmp);
        return vget_lane_f32(tmp, 0);
    #endif
    }

    inline batch<float, 4> haddp(const batch<float, 4>* row)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        float32x4_t tmp1 = vpaddq_f32(row[0], row[1]);
        float32x4_t tmp2 = vpaddq_f32(row[2], row[3]);
        return vpaddq_f32(tmp1, tmp2);
    #else
        // row = (a,b,c,d)
        float32x2_t tmp1, tmp2, tmp3;
        // tmp1 = (a0 + a2, a1 + a3)
        tmp1 = vpadd_f32(vget_low_f32(row[0]), vget_high_f32(row[0]));
        // tmp2 = (b0 + b2, b1 + b3)
        tmp2 = vpadd_f32(vget_low_f32(row[1]), vget_high_f32(row[1]));
        // tmp1 = (a0..3, b0..3)
        tmp1 = vpadd_f32(tmp1, tmp2);
        // tmp2 = (c0 + c2, c1 + c3)
        tmp2 = vpadd_f32(vget_low_f32(row[2]), vget_high_f32(row[2]));
        // tmp3 = (d0 + d2, d1 + d3)
        tmp3 = vpadd_f32(vget_low_f32(row[3]), vget_high_f32(row[3]));
        // tmp1 = (c0..3, d0..3)
        tmp2 = vpadd_f32(tmp2, tmp3);
        // return = (a0..3, b0..3, c0..3, d0..3)
        return vcombine_f32(tmp1, tmp2);
    #endif
    }

    inline batch_bool<float, 4> isnan(const batch<float, 4>& x)
    {
        return !(x == x);
    }

    inline batch_bool<float, 4> operator==(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vceqq_f32(lhs, rhs);
    }

    inline batch_bool<float, 4> operator!=(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return !(lhs == rhs);
    }

    inline batch_bool<float, 4> operator<(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vcltq_f32(lhs, rhs);
    }

    inline batch_bool<float, 4> operator<=(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vcleq_f32(lhs, rhs);
    }

    inline batch_bool<float, 4> operator>(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vcgtq_f32(lhs, rhs);
    }

    inline batch_bool<float, 4> operator>=(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vcgeq_f32(lhs, rhs);
    }

    inline batch<float, 4> select(const batch_bool<float, 4>& cond, const batch<float, 4>& a, const batch<float, 4>& b)
    {
        return vbslq_f32(cond, a, b);
    }

    inline batch<float, 4> operator&(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(lhs),
                                               vreinterpretq_u32_f32(rhs)));
    }

    inline batch<float, 4> operator|(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(lhs),
                                               vreinterpretq_u32_f32(rhs)));
    }

    inline batch<float, 4> operator^(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(lhs),
                                               vreinterpretq_u32_f32(rhs)));
    }

    inline batch<float, 4> operator~(const batch<float, 4>& rhs)
    {
        return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(rhs)));
    }

    inline batch<float, 4> bitwise_andnot(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
    }
}

#endif
