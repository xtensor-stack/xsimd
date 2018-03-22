/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_NEON_DOUBLE_HPP
#define XSIMD_NEON_DOUBLE_HPP

#include "xsimd_base.hpp"

namespace xsimd
{
    template <>
    struct simd_batch_traits<batch<double, 2>>
    {
        using value_type = double;
        static constexpr std::size_t size = 2;
        using batch_bool_type = batch_bool<double, 2>;
        static constexpr std::size_t align = XSIMD_DEFAULT_ALIGNMENT;
    };

    template <>
    class batch<double, 2> : public simd_batch<batch<double, 2>>
    {
        using simd_type = float64x2_t;

    public:

        batch();
        explicit batch(double d);
        batch(double d0, double d1);
        explicit batch(const double* src);

        batch(const double* src, aligned_mode);
        batch(const double* src, unaligned_mode);

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

        double operator[](std::size_t index) const;

    private:

        simd_type m_value;
    };

    /**
     * Implementation of batch
     */

    inline batch<double, 2>::batch()
    {
    }

    inline batch<double, 2>::batch(double d)
        : m_value(vdupq_n_f64(d))
    {
    }

    inline batch<double, 2>::batch(double d1, double d2)
        : m_value{d1, d2}
    {
    }

    inline batch<double, 2>::batch(const double* d)
        : m_value(vld1q_f64(d))
    {
    }

    inline batch<double, 2>::batch(const double* d, aligned_mode)
        : batch(d)
    {
    }

    inline batch<double, 2>::batch(const double* d, unaligned_mode)
        : batch(d)
    {
    }

    inline batch<double, 2>::batch(const simd_type& rhs)
        : m_value(rhs)
    {
    }

    inline batch<double, 2>& batch<double, 2>::operator=(const simd_type& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<double, 2>& batch<double, 2>::load_aligned(const float* d)
    {
        m_value = vcvt_f64_f32(vld1_f32(d));
        return *this;
    }

    inline batch<double, 2>& batch<double, 2>::load_unaligned(const float* d)
    {
        return load_unaligned(d);
    }

    inline batch<double, 2>& batch<double, 2>::load_aligned(const double* d)
    {
        m_value = vld1q_f64(d);
        return *this;
    }

    inline batch<double, 2>& batch<double, 2>::load_unaligned(const double* d)
    {
        return load_aligned(d);
    }

    inline batch<double, 2>& batch<double, 2>::load_aligned(const int32_t* d)
    {
        m_value = vcvt_f64_f32(vcvt_f32_s32(vld1_s32(d)));
        return *this;
    }

    inline batch<double, 2>& batch<double, 2>::load_unaligned(const int32_t* d)
    {
        return load_aligned(d);
    }

    inline batch<double, 2>& batch<double, 2>::load_aligned(const int64_t* d)
    {
        m_value = vcvtq_f64_s64(vld1q_s64(d));
        return *this;
    }

    inline batch<double, 2>& batch<double, 2>::load_unaligned(const int64_t* d)
    {
        return load_aligned(d);
    }

    inline void batch<double, 2>::store_aligned(float* dst) const
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        vst1_f32(dst, vcvt_f32_f64(m_value));
    #else
        dst[0] = static_cast<float>(m_value[0]);
        dst[1] = static_cast<float>(m_value[1]);
    #endif
    }

    inline void batch<double, 2>::store_unaligned(float* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<double, 2>::store_aligned(double* dst) const
    {
        vst1q_f64(dst, m_value);
    }

    inline void batch<double, 2>::store_unaligned(double* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<double, 2>::store_aligned(int32_t* dst) const
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        vst1_s32(dst, vcvt_s32_f32(vcvt_f32_f64(m_value)));
    #else
        dst[0] = static_cast<int32_t>(m_value[0]);
        dst[1] = static_cast<int32_t>(m_value[1]);
    #endif
    }

    inline void batch<double, 2>::store_unaligned(int32_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<double, 2>::store_aligned(int64_t* dst) const
    {
        vst1q_s64(dst, vcvtq_s64_f64(m_value));
    }

    inline void batch<double, 2>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline batch<double, 2>::operator float64x2_t() const
    {
        return m_value;
    }

    inline double batch<double, 2>::operator[](std::size_t index) const
    {
        return m_value[index];
    }

    inline batch<double, 2> operator-(const batch<double, 2>& lhs)
    {
        return vnegq_f64(lhs);
    }

    inline batch<double, 2> operator+(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return vaddq_f64(lhs, rhs);
    }

    inline batch<double, 2> operator-(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return vsubq_f64(lhs, rhs);
    }

    inline batch<double, 2> operator*(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return vmulq_f64(lhs, rhs);
    }

    inline batch<double, 2> operator/(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        return vdivq_f64(lhs, rhs);
    #else
        // from stackoverflow & https://projectne10.github.io/Ne10/doc/NE10__divc_8neon_8c_source.html
        // get an initial estimate of 1/b.
        float64x2_t reciprocal = vrecpeq_f64(rhs);

        // use a couple Newton-Raphson steps to refine the estimate.  Depending on your
        // application's accuracy requirements, you may be able to get away with only
        // one refinement (instead of the two used here).  Be sure to test!
        reciprocal = vmulq_f64(vrecpsq_f64(rhs, reciprocal), reciprocal);
        reciprocal = vmulq_f64(vrecpsq_f64(rhs, reciprocal), reciprocal);

        // and finally, compute a / b = a * (1 / b)
        return vmulq_f64(lhs, reciprocal);
    #endif
    }

    inline batch<double, 2> min(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return vminq_f64(lhs, rhs);
    }

    inline batch<double, 2> max(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return vmaxq_f64(lhs, rhs);
    }

    inline batch<double, 2> fmin(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return min(lhs, rhs);
    }

    inline batch<double, 2> fmax(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return max(lhs, rhs);
    }

    inline batch<double, 2> abs(const batch<double, 2>& lhs)
    {
        return vabsq_f64(lhs);
    }

    inline batch<double, 2> fabs(const batch<double, 2>& lhs)
    {
        return abs(lhs);
    }

    inline batch<double, 2> sqrt(const batch<double, 2>& lhs)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        return vsqrtq_f64(lhs);
    #else
        float64x2_t sqrt_reciprocal = vrsqrteq_f64(lhs);
        // one iter
        // sqrt_reciprocal = sqrt_reciprocal * vrsqrtsq_f64(lhs * sqrt_reciprocal, sqrt_reciprocal);
        return lhs * sqrt_reciprocal * vrsqrtsq_f64(lhs * sqrt_reciprocal, sqrt_reciprocal);
    #endif
    }

    inline batch<double, 2> fma(const batch<double, 2>& x, const batch<double, 2>& y, const batch<double, 2>& z)
    {
        return vfmaq_f64(z, x, y);
    }

    inline batch<double, 2> fms(const batch<double, 2>& x, const batch<double, 2>& y, const batch<double, 2>& z)
    {
        // return vfmsq_f64(-z, x, y);
        return vfmaq_f64(-z, x, y);
    }

    inline batch<double, 2> fnma(const batch<double, 2>& x, const batch<double, 2>& y, const batch<double, 2>& z)
    {
        return fma(-x, y, z);
    }

    inline batch<double, 2> fnms(const batch<double, 2>& x, const batch<double, 2>& y, const batch<double, 2>& z)
    {
        return fms(-x, y, z);
    }

    inline double hadd(const batch<double, 2>& rhs)
    {
    #if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
        return vaddvq_f64(rhs);
    #else
        float64x2_t tmp = vpaddq_f64(rhs, rhs);
        return vgetq_lane_f64(tmp, 0);
    #endif
    }

    inline batch<double, 2> haddp(const batch<double, 2>* row)
    {
        return vpaddq_f64(row[0], row[1]);
    }

    inline batch_bool<double, 2> operator==(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return vceqq_f64(lhs, rhs);
    }

    inline batch_bool<double, 2> operator!=(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return !(lhs == rhs);
    }

    inline batch_bool<double, 2> isnan(const batch<double, 2>& x)
    {
        return !(x == x);
    }

    inline batch_bool<double, 2> operator<(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return vcltq_f64(lhs, rhs);
    }

    inline batch_bool<double, 2> operator<=(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return vcleq_f64(lhs, rhs);
    }

    inline batch_bool<double, 2> operator>(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return vcgtq_f64(lhs, rhs);
    }

    inline batch_bool<double, 2> operator>=(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return vcgeq_f64(lhs, rhs);
    }

    inline batch<double, 2> select(const batch_bool<double, 2>& cond, const batch<double, 2>& a, const batch<double, 2>& b)
    {
        return vbslq_f64(cond, a, b);
    }

    inline batch<double, 2> operator&(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(lhs),
                                               vreinterpretq_u64_f64(rhs)));
    }

    inline batch<double, 2> operator|(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(lhs),
                                               vreinterpretq_u64_f64(rhs)));
    }

    inline batch<double, 2> operator^(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(lhs),
                                               vreinterpretq_u64_f64(rhs)));
    }

    inline batch<double, 2> operator~(const batch<double, 2>& rhs)
    {
        return vreinterpretq_f64_u32(vmvnq_u32(vreinterpretq_u32_f64(rhs)));
    }
}

#endif
