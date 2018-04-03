/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX_DOUBLE_HPP
#define XSIMD_AVX_DOUBLE_HPP

#include "xsimd_base.hpp"

namespace xsimd
{

    /*************************
     * batch_bool<double, 4> *
     *************************/

    template <>
    struct simd_batch_traits<batch_bool<double, 4>>
    {
        using value_type = bool;
        static constexpr std::size_t size = 4;
        using batch_type = batch<double, 4>;
        static constexpr std::size_t align = 32;
    };

    template <>
    class batch_bool<double, 4> : public simd_batch_bool<batch_bool<double, 4>>
    {
    public:

        batch_bool();
        explicit batch_bool(bool b);
        batch_bool(bool b0, bool b1, bool b2, bool b3);
        batch_bool(const __m256d& rhs);
        batch_bool& operator=(const __m256d& rhs);

        operator __m256d() const;

    private:

        __m256d m_value;
    };

    batch_bool<double, 4> operator&(const batch_bool<double, 4>& lhs, const batch_bool<double, 4>& rhs);
    batch_bool<double, 4> operator|(const batch_bool<double, 4>& lhs, const batch_bool<double, 4>& rhs);
    batch_bool<double, 4> operator^(const batch_bool<double, 4>& lhs, const batch_bool<double, 4>& rhs);
    batch_bool<double, 4> operator~(const batch_bool<double, 4>& rhs);
    batch_bool<double, 4> operator|(const batch_bool<double, 4>& lhs, const batch_bool<double, 4>& rhs);
    batch_bool<double, 4> bitwise_andnot(const batch_bool<double, 4>& lhs, const batch_bool<double, 4>& rhs);

    batch_bool<double, 4> operator==(const batch_bool<double, 4>& lhs, const batch_bool<double, 4>& rhs);
    batch_bool<double, 4> operator!=(const batch_bool<double, 4>& lhs, const batch_bool<double, 4>& rhs);

    bool all(const batch_bool<double, 4>& rhs);
    bool any(const batch_bool<double, 4>& rhs);

    /********************
     * batch<double, 4> *
     ********************/

    template <>
    struct simd_batch_traits<batch<double, 4>>
    {
        using value_type = double;
        static constexpr std::size_t size = 4;
        using batch_bool_type = batch_bool<double, 4>;
        static constexpr std::size_t align = 32;
    };

    template <>
    class batch<double, 4> : public simd_batch<batch<double, 4>>
    {
    public:

        batch();
        explicit batch(double d);
        batch(double d0, double d1, double d2, double d3);
        explicit batch(const double* src);
        batch(const double* src, aligned_mode);
        batch(const double* src, unaligned_mode);
        batch(const __m256d& rhs);
        batch& operator=(const __m256d& rhs);

        operator __m256d() const;

        batch& load_aligned(const double* src);
        batch& load_unaligned(const double* src);

        batch& load_aligned(const float* src);
        batch& load_unaligned(const float* src);

        batch& load_aligned(const int32_t* src);
        batch& load_unaligned(const int32_t* src);

        batch& load_aligned(const int64_t* src);
        batch& load_unaligned(const int64_t* src);

        void store_aligned(double* dst) const;
        void store_unaligned(double* dst) const;

        void store_aligned(float* dst) const;
        void store_unaligned(float* dst) const;

        void store_aligned(int32_t* dst) const;
        void store_unaligned(int32_t* dst) const;

        void store_aligned(int64_t* dst) const;
        void store_unaligned(int64_t* dst) const;

        double operator[](std::size_t index) const;

    private:

        __m256d m_value;
    };

    batch<double, 4> operator-(const batch<double, 4>& rhs);
    batch<double, 4> operator+(const batch<double, 4>& lhs, const batch<double, 4>& rhs);
    batch<double, 4> operator-(const batch<double, 4>& lhs, const batch<double, 4>& rhs);
    batch<double, 4> operator*(const batch<double, 4>& lhs, const batch<double, 4>& rhs);
    batch<double, 4> operator/(const batch<double, 4>& lhs, const batch<double, 4>& rhs);

    batch_bool<double, 4> operator==(const batch<double, 4>& lhs, const batch<double, 4>& rhs);
    batch_bool<double, 4> operator!=(const batch<double, 4>& lhs, const batch<double, 4>& rhs);
    batch_bool<double, 4> operator<(const batch<double, 4>& lhs, const batch<double, 4>& rhs);
    batch_bool<double, 4> operator<=(const batch<double, 4>& lhs, const batch<double, 4>& rhs);

    batch<double, 4> operator&(const batch<double, 4>& lhs, const batch<double, 4>& rhs);
    batch<double, 4> operator|(const batch<double, 4>& lhs, const batch<double, 4>& rhs);
    batch<double, 4> operator^(const batch<double, 4>& lhs, const batch<double, 4>& rhs);
    batch<double, 4> operator~(const batch<double, 4>& rhs);
    batch<double, 4> bitwise_andnot(const batch<double, 4>& lhs, const batch<double, 4>& rhs);

    batch<double, 4> min(const batch<double, 4>& lhs, const batch<double, 4>& rhs);
    batch<double, 4> max(const batch<double, 4>& lhs, const batch<double, 4>& rhs);
    batch<double, 4> fmin(const batch<double, 4>& lhs, const batch<double, 4>& rhs);
    batch<double, 4> fmax(const batch<double, 4>& lhs, const batch<double, 4>& rhs);

    batch<double, 4> abs(const batch<double, 4>& rhs);
    batch<double, 4> fabs(const batch<double, 4>& rhs);
    batch<double, 4> sqrt(const batch<double, 4>& rhs);

    batch<double, 4> fma(const batch<double, 4>& x, const batch<double, 4>& y, const batch<double, 4>& z);
    batch<double, 4> fms(const batch<double, 4>& x, const batch<double, 4>& y, const batch<double, 4>& z);
    batch<double, 4> fnma(const batch<double, 4>& x, const batch<double, 4>& y, const batch<double, 4>& z);
    batch<double, 4> fnms(const batch<double, 4>& x, const batch<double, 4>& y, const batch<double, 4>& z);

    double hadd(const batch<double, 4>& rhs);
    batch<double, 4> haddp(const batch<double, 4>* row);

    batch<double, 4> select(const batch_bool<double, 4>& cond, const batch<double, 4>& a, const batch<double, 4>& b);

    batch_bool<double, 4> isnan(const batch<double, 4>& x);

    /****************************************
     * batch_bool<double, 4> implementation *
     ****************************************/

    inline batch_bool<double, 4>::batch_bool()
    {
    }

    inline batch_bool<double, 4>::batch_bool(bool b)
        : m_value(_mm256_castsi256_pd(_mm256_set1_epi32(-(int)b)))
    {
    }

    inline batch_bool<double, 4>::batch_bool(bool b0, bool b1, bool b2, bool b3)
        : m_value(_mm256_castsi256_pd(
              _mm256_setr_epi32(-(int)b0, -(int)b0, -(int)b1, -(int)b1,
                                -(int)b2, -(int)b2, -(int)b3, -(int)b3)))
    {
    }

    inline batch_bool<double, 4>::batch_bool(const __m256d& rhs)
        : m_value(rhs)
    {
    }

    inline batch_bool<double, 4>& batch_bool<double, 4>::operator=(const __m256d& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch_bool<double, 4>::operator __m256d() const
    {
        return m_value;
    }

    inline batch_bool<double, 4> operator&(const batch_bool<double, 4>& lhs, const batch_bool<double, 4>& rhs)
    {
        return _mm256_and_pd(lhs, rhs);
    }

    inline batch_bool<double, 4> operator|(const batch_bool<double, 4>& lhs, const batch_bool<double, 4>& rhs)
    {
        return _mm256_or_pd(lhs, rhs);
    }

    inline batch_bool<double, 4> operator^(const batch_bool<double, 4>& lhs, const batch_bool<double, 4>& rhs)
    {
        return _mm256_xor_pd(lhs, rhs);
    }

    inline batch_bool<double, 4> operator~(const batch_bool<double, 4>& rhs)
    {
        return _mm256_xor_pd(rhs, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
    }

    inline batch_bool<double, 4> bitwise_andnot(const batch_bool<double, 4>& lhs, const batch_bool<double, 4>& rhs)
    {
        return _mm256_andnot_pd(lhs, rhs);
    }

    inline batch_bool<double, 4> operator==(const batch_bool<double, 4>& lhs, const batch_bool<double, 4>& rhs)
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_EQ_OQ);
    }

    inline batch_bool<double, 4> operator!=(const batch_bool<double, 4>& lhs, const batch_bool<double, 4>& rhs)
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_NEQ_OQ);
    }

    inline bool all(const batch_bool<double, 4>& rhs)
    {
        return _mm256_testc_pd(rhs, batch_bool<double, 4>(true)) != 0;
    }

    inline bool any(const batch_bool<double, 4>& rhs)
    {
        return !_mm256_testz_pd(rhs, rhs);
    }

    /***********************************
     * batch<double, 4> implementation *
     ***********************************/

    inline batch<double, 4>::batch()
    {
    }

    inline batch<double, 4>::batch(double d)
        : m_value(_mm256_set1_pd(d))
    {
    }

    inline batch<double, 4>::batch(double d0, double d1, double d2, double d3)
        : m_value(_mm256_setr_pd(d0, d1, d2, d3))
    {
    }

    inline batch<double, 4>::batch(const double* src)
        : m_value(_mm256_loadu_pd(src))
    {
    }

    inline batch<double, 4>::batch(const double* src, aligned_mode)
        : m_value(_mm256_load_pd(src))
    {
    }

    inline batch<double, 4>::batch(const double* src, unaligned_mode)
        : m_value(_mm256_loadu_pd(src))
    {
    }

    inline batch<double, 4>::batch(const __m256d& rhs)
        : m_value(rhs)
    {
    }

    inline batch<double, 4>& batch<double, 4>::operator=(const __m256d& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<double, 4>::operator __m256d() const
    {
        return m_value;
    }

    inline batch<double, 4>& batch<double, 4>::load_aligned(const double* src)
    {
        m_value = _mm256_load_pd(src);
        return *this;
    }

    inline batch<double, 4>& batch<double, 4>::load_unaligned(const double* src)
    {
        m_value = _mm256_loadu_pd(src);
        return *this;
    }

    inline batch<double, 4>& batch<double, 4>::load_aligned(const float* src)
    {
        m_value = _mm256_cvtps_pd(_mm_load_ps(src));
        return *this;
    }

    inline batch<double, 4>& batch<double, 4>::load_unaligned(const float* src)
    {
        m_value = _mm256_cvtps_pd(_mm_loadu_ps(src));
        return *this;
    }

    inline batch<double, 4>& batch<double, 4>::load_aligned(const int32_t* src)
    {
        m_value = _mm256_cvtepi32_pd(_mm_load_si128((__m128i const*)src));
        return *this;
    }

    inline batch<double, 4>& batch<double, 4>::load_unaligned(const int32_t* src)
    {
        m_value = _mm256_cvtepi32_pd(_mm_loadu_si128((__m128i const*)src));
        return *this;
    }

    inline batch<double, 4>& batch<double, 4>::load_aligned(const int64_t* src)
    {
        alignas(32) double tmp[4];
        tmp[0] = double(src[0]);
        tmp[1] = double(src[1]);
        tmp[2] = double(src[2]);
        tmp[3] = double(src[3]);
        m_value = _mm256_load_pd(tmp);
        return *this;
    }

    inline batch<double, 4>& batch<double, 4>::load_unaligned(const int64_t* src)
    {
        return load_aligned(src);
    }

    inline void batch<double, 4>::store_aligned(double* dst) const
    {
        _mm256_store_pd(dst, m_value);
    }

    inline void batch<double, 4>::store_unaligned(double* dst) const
    {
        _mm256_storeu_pd(dst, m_value);
    }

    inline void batch<double, 4>::store_aligned(float* dst) const
    {
        _mm_store_ps(dst, _mm256_cvtpd_ps(m_value));
    }

    inline void batch<double, 4>::store_unaligned(float* dst) const
    {
        _mm_storeu_ps(dst, _mm256_cvtpd_ps(m_value));
    }

    inline void batch<double, 4>::store_aligned(int32_t* dst) const
    {
        _mm_store_si128((__m128i*)dst, _mm256_cvtpd_epi32(m_value));
    }

    inline void batch<double, 4>::store_unaligned(int32_t* dst) const
    {
        _mm_storeu_si128((__m128i*)dst, _mm256_cvtpd_epi32(m_value));
    }

    inline void batch<double, 4>::store_aligned(int64_t* dst) const
    {
        alignas(32) double tmp[4];
        _mm256_store_pd(tmp, m_value);
        dst[0] = static_cast<int64_t>(tmp[0]);
        dst[1] = static_cast<int64_t>(tmp[1]);
        dst[2] = static_cast<int64_t>(tmp[2]);
        dst[3] = static_cast<int64_t>(tmp[3]);
    }

    inline void batch<double, 4>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline double batch<double, 4>::operator[](std::size_t index) const
    {
        alignas(32) double x[4];
        store_aligned(x);
        return x[index & 3];
    }

    inline batch<double, 4> operator-(const batch<double, 4>& rhs)
    {
        return _mm256_xor_pd(rhs, _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000)));
    }

    inline batch<double, 4> operator+(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_add_pd(lhs, rhs);
    }

    inline batch<double, 4> operator-(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_sub_pd(lhs, rhs);
    }

    inline batch<double, 4> operator*(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_mul_pd(lhs, rhs);
    }

    inline batch<double, 4> operator/(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_div_pd(lhs, rhs);
    }

    inline batch_bool<double, 4> operator==(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_EQ_OQ);
    }

    inline batch_bool<double, 4> operator!=(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_NEQ_OQ);
    }

    inline batch_bool<double, 4> operator<(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_LT_OQ);
    }

    inline batch_bool<double, 4> operator<=(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_LE_OQ);
    }

    inline batch<double, 4> operator&(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_and_pd(lhs, rhs);
    }

    inline batch<double, 4> operator|(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_or_pd(lhs, rhs);
    }

    inline batch<double, 4> operator^(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_xor_pd(lhs, rhs);
    }

    inline batch<double, 4> operator~(const batch<double, 4>& rhs)
    {
        return _mm256_xor_pd(rhs, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
    }

    inline batch<double, 4> bitwise_andnot(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_andnot_pd(lhs, rhs);
    }

    inline batch<double, 4> min(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_min_pd(lhs, rhs);
    }

    inline batch<double, 4> max(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return _mm256_max_pd(lhs, rhs);
    }

    inline batch<double, 4> fmin(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return min(lhs, rhs);
    }

    inline batch<double, 4> fmax(const batch<double, 4>& lhs, const batch<double, 4>& rhs)
    {
        return max(lhs, rhs);
    }

    inline batch<double, 4> abs(const batch<double, 4>& rhs)
    {
        __m256d sign_mask = _mm256_set1_pd(-0.);  // -0. = 1 << 63
        return _mm256_andnot_pd(sign_mask, rhs);
    }

    inline batch<double, 4> fabs(const batch<double, 4>& rhs)
    {
        return abs(rhs);
    }

    inline batch<double, 4> sqrt(const batch<double, 4>& rhs)
    {
        return _mm256_sqrt_pd(rhs);
    }

    inline batch<double, 4> fma(const batch<double, 4>& x, const batch<double, 4>& y, const batch<double, 4>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm256_fmadd_pd(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm256_macc_pd(x, y, z);
#else
        return x * y + z;
#endif
    }

    inline batch<double, 4> fms(const batch<double, 4>& x, const batch<double, 4>& y, const batch<double, 4>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm256_fmsub_pd(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm256_msub_pd(x, y, z);
#else
        return x * y - z;
#endif
    }

    inline batch<double, 4> fnma(const batch<double, 4>& x, const batch<double, 4>& y, const batch<double, 4>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm256_fnmadd_pd(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm256_nmacc_pd(x, y, z);
#else
        return -x * y + z;
#endif
    }

    inline batch<double, 4> fnms(const batch<double, 4>& x, const batch<double, 4>& y, const batch<double, 4>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm256_fnmsub_pd(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm256_nmsub_pd(x, y, z);
#else
        return -x * y - z;
#endif
    }

    inline double hadd(const batch<double, 4>& rhs)
    {
        // rhs = (x0, x1, x2, x3)
        // tmp = (x2, x3, x0, x1)
        __m256d tmp = _mm256_permute2f128_pd(rhs, rhs, 1);
        // tmp = (x2+x0, x3+x1, -, -)
        tmp = _mm256_add_pd(rhs, tmp);
        // tmp = (x2+x0+x3+x1, -, -, -)
        tmp = _mm256_hadd_pd(tmp, tmp);
        return _mm_cvtsd_f64(_mm256_extractf128_pd(tmp, 0));
    }

    inline batch<double, 4> haddp(const batch<double, 4>* row)
    {
        // row = (a,b,c,d)
        // tmp0 = (a0+a1, b0+b1, a2+a3, b2+b3)
        __m256d tmp0 = _mm256_hadd_pd(row[0], row[1]);
        // tmp1 = (c0+c1, d0+d1, c2+c3, d2+d3)
        __m256d tmp1 = _mm256_hadd_pd(row[2], row[3]);
        // tmp2 = (a0+a1, b0+b1, c2+c3, d2+d3)
        __m256d tmp2 = _mm256_blend_pd(tmp0, tmp1, 0b1100);
        // tmp1 = (a2+a3, b2+b3, c2+c3, d2+d3)
        tmp1 = _mm256_permute2f128_pd(tmp0, tmp1, 0x21);
        return _mm256_add_pd(tmp1, tmp2);
    }

    inline batch<double, 4> select(const batch_bool<double, 4>& cond, const batch<double, 4>& a, const batch<double, 4>& b)
    {
        return _mm256_blendv_pd(b, a, cond);
    }

    inline batch_bool<double, 4> isnan(const batch<double, 4>& x)
    {
        return _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
    }
}

#endif
