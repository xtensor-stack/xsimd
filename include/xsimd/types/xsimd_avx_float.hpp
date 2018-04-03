/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX_FLOAT_HPP
#define XSIMD_AVX_FLOAT_HPP

#include "xsimd_base.hpp"

namespace xsimd
{

    /************************
     * batch_bool<float, 8> *
     ************************/

    template <>
    struct simd_batch_traits<batch_bool<float, 8>>
    {
        using value_type = bool;
        static constexpr std::size_t size = 8;
        using batch_type = batch<float, 8>;
        static constexpr std::size_t align = 32;
    };

    template <>
    class batch_bool<float, 8> : public simd_batch_bool<batch_bool<float, 8>>
    {
    public:

        batch_bool();
        explicit batch_bool(bool b);
        batch_bool(bool b0, bool b1, bool b2, bool b3,
                   bool b4, bool b5, bool b6, bool b7);
        batch_bool(const __m256& rhs);
        batch_bool& operator=(const __m256& rhs);

        operator __m256() const;

    private:

        __m256 m_value;
    };

    batch_bool<float, 8> operator&(const batch_bool<float, 8>& lhs, const batch_bool<float, 8>& rhs);
    batch_bool<float, 8> operator|(const batch_bool<float, 8>& lhs, const batch_bool<float, 8>& rhs);
    batch_bool<float, 8> operator^(const batch_bool<float, 8>& lhs, const batch_bool<float, 8>& rhs);
    batch_bool<float, 8> operator~(const batch_bool<float, 8>& rhs);
    batch_bool<float, 8> bitwise_andnot(const batch_bool<float, 8>& lhs, const batch_bool<float, 8>& rhs);

    batch_bool<float, 8> operator==(const batch_bool<float, 8>& lhs, const batch_bool<float, 8>& rhs);
    batch_bool<float, 8> operator!=(const batch_bool<float, 8>& lhs, const batch_bool<float, 8>& rhs);

    bool all(const batch_bool<float, 8>& rhs);
    bool any(const batch_bool<float, 8>& rhs);

    /*******************
     * batch<float, 8> *
     *******************/

    template <>
    struct simd_batch_traits<batch<float, 8>>
    {
        using value_type = float;
        static constexpr std::size_t size = 8;
        using batch_bool_type = batch_bool<float, 8>;
        static constexpr std::size_t align = 32;
    };

    template <>
    class batch<float, 8> : public simd_batch<batch<float, 8>>
    {
    public:

        batch();
        explicit batch(float f);
        batch(float f0, float f1, float f2, float f3,
              float f4, float f5, float f6, float f7);
        explicit batch(const float* src);
        batch(const float* src, aligned_mode);
        batch(const float* src, unaligned_mode);
        batch(const __m256& rhs);
        batch& operator=(const __m256& rhs);

        operator __m256() const;

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

        __m256 m_value;
    };

    batch<float, 8> operator-(const batch<float, 8>& rhs);
    batch<float, 8> operator+(const batch<float, 8>& lhs, const batch<float, 8>& rhs);
    batch<float, 8> operator-(const batch<float, 8>& lhs, const batch<float, 8>& rhs);
    batch<float, 8> operator*(const batch<float, 8>& lhs, const batch<float, 8>& rhs);
    batch<float, 8> operator/(const batch<float, 8>& lhs, const batch<float, 8>& rhs);

    batch_bool<float, 8> operator==(const batch<float, 8>& lhs, const batch<float, 8>& rhs);
    batch_bool<float, 8> operator!=(const batch<float, 8>& lhs, const batch<float, 8>& rhs);
    batch_bool<float, 8> operator<(const batch<float, 8>& lhs, const batch<float, 8>& rhs);
    batch_bool<float, 8> operator<=(const batch<float, 8>& lhs, const batch<float, 8>& rhs);

    batch<float, 8> operator&(const batch<float, 8>& lhs, const batch<float, 8>& rhs);
    batch<float, 8> operator|(const batch<float, 8>& lhs, const batch<float, 8>& rhs);
    batch<float, 8> operator^(const batch<float, 8>& lhs, const batch<float, 8>& rhs);
    batch<float, 8> operator~(const batch<float, 8>& rhs);
    batch<float, 8> bitwise_andnot(const batch<float, 8>& lhs, const batch<float, 8>& rhs);

    batch<float, 8> min(const batch<float, 8>& lhs, const batch<float, 8>& rhs);
    batch<float, 8> max(const batch<float, 8>& lhs, const batch<float, 8>& rhs);
    batch<float, 8> fmin(const batch<float, 8>& lhs, const batch<float, 8>& rhs);
    batch<float, 8> fmax(const batch<float, 8>& lhs, const batch<float, 8>& rhs);

    batch<float, 8> abs(const batch<float, 8>& rhs);
    batch<float, 8> fabs(const batch<float, 8>& rhs);
    batch<float, 8> sqrt(const batch<float, 8>& rhs);

    batch<float, 8> fma(const batch<float, 8>& x, const batch<float, 8>& y, const batch<float, 8>& z);
    batch<float, 8> fms(const batch<float, 8>& x, const batch<float, 8>& y, const batch<float, 8>& z);
    batch<float, 8> fnma(const batch<float, 8>& x, const batch<float, 8>& y, const batch<float, 8>& z);
    batch<float, 8> fnms(const batch<float, 8>& x, const batch<float, 8>& y, const batch<float, 8>& z);

    float hadd(const batch<float, 8>& rhs);
    batch<float, 8> haddp(const batch<float, 8>* row);

    batch<float, 8> select(const batch_bool<float, 8>& cond, const batch<float, 8>& a, const batch<float, 8>& b);

    batch_bool<float, 8> isnan(const batch<float, 8>& x);

    /***************************************
     * batch_bool<float, 8> implementation *
     ***************************************/

    inline batch_bool<float, 8>::batch_bool()
    {
    }

    inline batch_bool<float, 8>::batch_bool(bool b)
        : m_value(_mm256_castsi256_ps(_mm256_set1_epi32(-(int)b)))
    {
    }

    inline batch_bool<float, 8>::batch_bool(bool b0, bool b1, bool b2, bool b3,
                                            bool b4, bool b5, bool b6, bool b7)
        : m_value(_mm256_castsi256_ps(
              _mm256_setr_epi32(-(int)b0, -(int)b1, -(int)b2, -(int)b3,
                                -(int)b4, -(int)b5, -(int)b6, -(int)b7)))
    {
    }

    inline batch_bool<float, 8>::batch_bool(const __m256& rhs)
        : m_value(rhs)
    {
    }

    inline batch_bool<float, 8>& batch_bool<float, 8>::operator=(const __m256& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch_bool<float, 8>::operator __m256() const
    {
        return m_value;
    }

    inline batch_bool<float, 8> operator&(const batch_bool<float, 8>& lhs, const batch_bool<float, 8>& rhs)
    {
        return _mm256_and_ps(lhs, rhs);
    }

    inline batch_bool<float, 8> operator|(const batch_bool<float, 8>& lhs, const batch_bool<float, 8>& rhs)
    {
        return _mm256_or_ps(lhs, rhs);
    }

    inline batch_bool<float, 8> operator^(const batch_bool<float, 8>& lhs, const batch_bool<float, 8>& rhs)
    {
        return _mm256_xor_ps(lhs, rhs);
    }

    inline batch_bool<float, 8> operator~(const batch_bool<float, 8>& rhs)
    {
        return _mm256_xor_ps(rhs, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
    }

    inline batch_bool<float, 8> bitwise_andnot(const batch_bool<float, 8>& lhs, const batch_bool<float, 8>& rhs)
    {
        return _mm256_andnot_ps(lhs, rhs);
    }

    inline batch_bool<float, 8> operator==(const batch_bool<float, 8>& lhs, const batch_bool<float, 8>& rhs)
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_EQ_OQ);
    }

    inline batch_bool<float, 8> operator!=(const batch_bool<float, 8>& lhs, const batch_bool<float, 8>& rhs)
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NEQ_OQ);
    }

    inline bool all(const batch_bool<float, 8>& rhs)
    {
        return _mm256_testc_ps(rhs, batch_bool<float, 8>(true)) != 0;
    }

    inline bool any(const batch_bool<float, 8>& rhs)
    {
        return !_mm256_testz_ps(rhs, rhs);
    }

    /**********************************
     * batch<float, 8> implementation *
     **********************************/

    inline batch<float, 8>::batch()
    {
    }

    inline batch<float, 8>::batch(float f)
        : m_value(_mm256_set1_ps(f))
    {
    }

    inline batch<float, 8>::batch(float f0, float f1, float f2, float f3,
                                  float f4, float f5, float f6, float f7)
        : m_value(_mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7))
    {
    }

    inline batch<float, 8>::batch(const float* src)
        : m_value(_mm256_loadu_ps(src))
    {
    }

    inline batch<float, 8>::batch(const float* src, aligned_mode)
        : m_value(_mm256_load_ps(src))
    {
    }

    inline batch<float, 8>::batch(const float* src, unaligned_mode)
        : m_value(_mm256_loadu_ps(src))
    {
    }

    inline batch<float, 8>::batch(const __m256& rhs)
        : m_value(rhs)
    {
    }

    inline batch<float, 8>& batch<float, 8>::operator=(const __m256& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<float, 8>::operator __m256() const
    {
        return m_value;
    }

    inline batch<float, 8>& batch<float, 8>::load_aligned(const float* src)
    {
        m_value = _mm256_load_ps(src);
        return *this;
    }

    inline batch<float, 8>& batch<float, 8>::load_unaligned(const float* src)
    {
        m_value = _mm256_loadu_ps(src);
        return *this;
    }

    inline batch<float, 8>& batch<float, 8>::load_aligned(const double* src)
    {
        __m128 tmp1 = _mm256_cvtpd_ps(_mm256_load_pd(src));
        __m128 tmp2 = _mm256_cvtpd_ps(_mm256_load_pd(src + 4));
        m_value = _mm256_castps128_ps256(tmp1);
        m_value = _mm256_insertf128_ps(m_value, tmp2, 1);
        return *this;
    }

    inline batch<float, 8>& batch<float, 8>::load_unaligned(const double* src)
    {
        __m128 tmp1 = _mm256_cvtpd_ps(_mm256_loadu_pd(src));
        __m128 tmp2 = _mm256_cvtpd_ps(_mm256_loadu_pd(src + 4));
        m_value = _mm256_castps128_ps256(tmp1);
        m_value = _mm256_insertf128_ps(m_value, tmp2, 1);
        return *this;
    }

    inline batch<float, 8>& batch<float, 8>::load_aligned(const int32_t* src)
    {
        m_value = _mm256_cvtepi32_ps(_mm256_load_si256((__m256i const*)src));
        return *this;
    }

    inline batch<float, 8>& batch<float, 8>::load_unaligned(const int32_t* src)
    {
        m_value = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const*)src));
        return *this;
    }

    inline batch<float, 8>& batch<float, 8>::load_aligned(const int64_t* src)
    {
        alignas(32) float tmp[8];
        tmp[0] = float(src[0]);
        tmp[1] = float(src[1]);
        tmp[2] = float(src[2]);
        tmp[3] = float(src[3]);
        tmp[4] = float(src[4]);
        tmp[5] = float(src[5]);
        tmp[6] = float(src[6]);
        tmp[7] = float(src[7]);
        m_value = _mm256_load_ps(tmp);
        return *this;
    }

    inline batch<float, 8>& batch<float, 8>::load_unaligned(const int64_t* src)
    {
        return load_aligned(src);
    }

    inline void batch<float, 8>::store_aligned(float* dst) const
    {
        _mm256_store_ps(dst, m_value);
    }

    inline void batch<float, 8>::store_unaligned(float* dst) const
    {
        _mm256_storeu_ps(dst, m_value);
    }

    inline void batch<float, 8>::store_aligned(double* dst) const
    {
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, m_value);
        dst[0] = static_cast<double>(tmp[0]);
        dst[1] = static_cast<double>(tmp[1]);
        dst[2] = static_cast<double>(tmp[2]);
        dst[3] = static_cast<double>(tmp[3]);
        dst[4] = static_cast<double>(tmp[4]);
        dst[5] = static_cast<double>(tmp[5]);
        dst[6] = static_cast<double>(tmp[6]);
        dst[7] = static_cast<double>(tmp[7]);
    }

    inline void batch<float, 8>::store_unaligned(double* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<float, 8>::store_aligned(int32_t* dst) const
    {
        _mm256_store_si256((__m256i*)dst, _mm256_cvtps_epi32(m_value));
    }

    inline void batch<float, 8>::store_unaligned(int32_t* dst) const
    {
        _mm256_storeu_si256((__m256i*)dst, _mm256_cvtps_epi32(m_value));
    }

    inline void batch<float, 8>::store_aligned(int64_t* dst) const
    {
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, m_value);
        dst[0] = static_cast<int64_t>(tmp[0]);
        dst[1] = static_cast<int64_t>(tmp[1]);
        dst[2] = static_cast<int64_t>(tmp[2]);
        dst[3] = static_cast<int64_t>(tmp[3]);
        dst[4] = static_cast<int64_t>(tmp[4]);
        dst[5] = static_cast<int64_t>(tmp[5]);
        dst[6] = static_cast<int64_t>(tmp[6]);
        dst[7] = static_cast<int64_t>(tmp[7]);
    }

    inline void batch<float, 8>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline float batch<float, 8>::operator[](std::size_t index) const
    {
        alignas(32) float x[8];
        store_aligned(x);
        return x[index & 7];
    }

    inline batch<float, 8> operator-(const batch<float, 8>& rhs)
    {
        return _mm256_xor_ps(rhs, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
    }

    inline batch<float, 8> operator+(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_add_ps(lhs, rhs);
    }

    inline batch<float, 8> operator-(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_sub_ps(lhs, rhs);
    }

    inline batch<float, 8> operator*(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_mul_ps(lhs, rhs);
    }

    inline batch<float, 8> operator/(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_div_ps(lhs, rhs);
    }

    inline batch_bool<float, 8> operator==(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_EQ_OQ);
    }

    inline batch_bool<float, 8> operator!=(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NEQ_OQ);
    }

    inline batch_bool<float, 8> operator<(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ);
    }

    inline batch_bool<float, 8> operator<=(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_LE_OQ);
    }

    inline batch<float, 8> operator&(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_and_ps(lhs, rhs);
    }

    inline batch<float, 8> operator|(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_or_ps(lhs, rhs);
    }

    inline batch<float, 8> operator^(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_xor_ps(lhs, rhs);
    }

    inline batch<float, 8> operator~(const batch<float, 8>& rhs)
    {
        return _mm256_xor_ps(rhs, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
    }

    inline batch<float, 8> bitwise_andnot(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_andnot_ps(lhs, rhs);
    }

    inline batch<float, 8> min(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_min_ps(lhs, rhs);
    }

    inline batch<float, 8> max(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return _mm256_max_ps(lhs, rhs);
    }

    inline batch<float, 8> fmin(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return min(lhs, rhs);
    }

    inline batch<float, 8> fmax(const batch<float, 8>& lhs, const batch<float, 8>& rhs)
    {
        return max(lhs, rhs);
    }

    inline batch<float, 8> abs(const batch<float, 8>& rhs)
    {
        __m256 sign_mask = _mm256_set1_ps(-0.f);  // -0.f = 1 << 31
        return _mm256_andnot_ps(sign_mask, rhs);
    }

    inline batch<float, 8> fabs(const batch<float, 8>& rhs)
    {
        return abs(rhs);
    }

    inline batch<float, 8> sqrt(const batch<float, 8>& rhs)
    {
        return _mm256_sqrt_ps(rhs);
    }

    inline batch<float, 8> fma(const batch<float, 8>& x, const batch<float, 8>& y, const batch<float, 8>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm256_fmadd_ps(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm256_macc_ps(x, y, z);
#else
        return x * y + z;
#endif
    }

    inline batch<float, 8> fms(const batch<float, 8>& x, const batch<float, 8>& y, const batch<float, 8>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm256_fmsub_ps(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm256_msub_ps(x, y, z);
#else
        return x * y - z;
#endif
    }

    inline batch<float, 8> fnma(const batch<float, 8>& x, const batch<float, 8>& y, const batch<float, 8>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm256_fnmadd_ps(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm256_nmacc_ps(x, y, z);
#else
        return -x * y + z;
#endif
    }

    inline batch<float, 8> fnms(const batch<float, 8>& x, const batch<float, 8>& y, const batch<float, 8>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm256_fnmsub_ps(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm256_nmsub_ps(x, y, z);
#else
        return -x * y - z;
#endif
    }

    inline float hadd(const batch<float, 8>& rhs)
    {
        // Warning about _mm256_hadd_ps:
        // _mm256_hadd_ps(a,b) gives
        // (a0+a1,a2+a3,b0+b1,b2+b3,a4+a5,a6+a7,b4+b5,b6+b7). Hence we can't
        // rely on a naive use of this method
        // rhs = (x0, x1, x2, x3, x4, x5, x6, x7)
        // tmp = (x4, x5, x6, x7, x0, x1, x2, x3)
        __m256 tmp = _mm256_permute2f128_ps(rhs, rhs, 1);
        // tmp = (x4+x0, x5+x1, x6+x2, x7+x3, x0+x4, x1+x5, x2+x6, x3+x7)
        tmp = _mm256_add_ps(rhs, tmp);
        // tmp = (x4+x0+x5+x1, x6+x2+x7+x3, -, -, -, -, -, -)
        tmp = _mm256_hadd_ps(tmp, tmp);
        // tmp = (x4+x0+x5+x1+x6+x2+x7+x3, -, -, -, -, -, -, -)
        tmp = _mm256_hadd_ps(tmp, tmp);
        return _mm_cvtss_f32(_mm256_extractf128_ps(tmp, 0));
    }

    inline batch<float, 8> haddp(const batch<float, 8>* row)
    {
        // row = (a,b,c,d,e,f,g,h)
        // tmp0 = (a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7)
        __m256 tmp0 = _mm256_hadd_ps(row[0], row[1]);
        // tmp1 = (c0+c1, c2+c3, d1+d2, d2+d3, c4+c5, c6+c7, d4+d5, d6+d7)
        __m256 tmp1 = _mm256_hadd_ps(row[2], row[3]);
        // tmp1 = (a0+a1+a2+a3, b0+b1+b2+b3, c0+c1+c2+c3, d0+d1+d2+d3,
        // a4+a5+a6+a7, b4+b5+b6+b7, c4+c5+c6+c7, d4+d5+d6+d7)
        tmp1 = _mm256_hadd_ps(tmp0, tmp1);
        // tmp0 = (e0+e1, e2+e3, f0+f1, f2+f3, e4+e5, e6+e7, f4+f5, f6+f7)
        tmp0 = _mm256_hadd_ps(row[4], row[5]);
        // tmp2 = (g0+g1, g2+g3, h0+h1, h2+h3, g4+g5, g6+g7, h4+h5, h6+h7)
        __m256 tmp2 = _mm256_hadd_ps(row[6], row[7]);
        // tmp2 = (e0+e1+e2+e3, f0+f1+f2+f3, g0+g1+g2+g3, h0+h1+h2+h3,
        // e4+e5+e6+e7, f4+f5+f6+f7, g4+g5+g6+g7, h4+h5+h6+h7)
        tmp2 = _mm256_hadd_ps(tmp0, tmp2);
        // tmp0 = (a0+a1+a2+a3, b0+b1+b2+b3, c0+c1+c2+c3, d0+d1+d2+d3,
        // e4+e5+e6+e7, f4+f5+f6+f7, g4+g5+g6+g7, h4+h5+h6+h7)
        tmp0 = _mm256_blend_ps(tmp1, tmp2, 0b11110000);
        // tmp1 = (a4+a5+a6+a7, b4+b5+b6+b7, c4+c5+c6+c7, d4+d5+d6+d7,
        // e0+e1+e2+e3, f0+f1+f2+f3, g0+g1+g2+g3, h0+h1+h2+h3)
        tmp1 = _mm256_permute2f128_ps(tmp1, tmp2, 0x21);
        return _mm256_add_ps(tmp0, tmp1);
    }

    inline batch<float, 8> select(const batch_bool<float, 8>& cond, const batch<float, 8>& a, const batch<float, 8>& b)
    {
        return _mm256_blendv_ps(b, a, cond);
    }

    inline batch_bool<float, 8> isnan(const batch<float, 8>& x)
    {
        return _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    }
}

#endif
