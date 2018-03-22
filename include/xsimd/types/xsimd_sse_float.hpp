/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_SSE_FLOAT_HPP
#define XSIMD_SSE_FLOAT_HPP

#include "xsimd_base.hpp"

namespace xsimd
{

    /************************
     * batch_bool<float, 4> *
     ************************/

    template <>
    struct simd_batch_traits<batch_bool<float, 4>>
    {
        using value_type = bool;
        static constexpr std::size_t size = 4;
        using batch_type = batch<float, 4>;
        static constexpr std::size_t align = 16;
    };

    template <>
    class batch_bool<float, 4> : public simd_batch_bool<batch_bool<float, 4>>
    {
    public:

        batch_bool();
        explicit batch_bool(bool b);
        batch_bool(bool b0, bool b1, bool b2, bool b3);
        batch_bool(const __m128& rhs);
        batch_bool& operator=(const __m128& rhs);

        operator __m128() const;

    private:

        __m128 m_value;
    };

    batch_bool<float, 4> operator&(const batch_bool<float, 4>& lhs, const batch_bool<float, 4>& rhs);
    batch_bool<float, 4> operator|(const batch_bool<float, 4>& lhs, const batch_bool<float, 4>& rhs);
    batch_bool<float, 4> operator^(const batch_bool<float, 4>& lhs, const batch_bool<float, 4>& rhs);
    batch_bool<float, 4> operator~(const batch_bool<float, 4>& rhs);
    batch_bool<float, 4> bitwise_andnot(const batch_bool<float, 4>& lhs, const batch_bool<float, 4>& rhs);

    batch_bool<float, 4> operator==(const batch_bool<float, 4>& lhs, const batch_bool<float, 4>& rhs);
    batch_bool<float, 4> operator!=(const batch_bool<float, 4>& lhs, const batch_bool<float, 4>& rhs);

    bool all(const batch_bool<float, 4>& rhs);
    bool any(const batch_bool<float, 4>& rhs);

    /*******************
     * batch<float, 4> *
     *******************/

    template <>
    struct simd_batch_traits<batch<float, 4>>
    {
        using value_type = float;
        static constexpr std::size_t size = 4;
        using batch_bool_type = batch_bool<float, 4>;
        static constexpr std::size_t align = 16;
    };

    template <>
    class batch<float, 4> : public simd_batch<batch<float, 4>>
    {
    public:

        batch();
        explicit batch(float f);
        batch(float f0, float f1, float f2, float f3);
        explicit batch(const float* src);
        batch(const float* src, aligned_mode);
        batch(const float* src, unaligned_mode);
        batch(const __m128& rhs);
        batch& operator=(const __m128& rhs);

        operator __m128() const;

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

        __m128 m_value;
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

    /***************************************
     * batch_bool<float, 4> implementation *
     ***************************************/

    inline batch_bool<float, 4>::batch_bool()
    {
    }

    inline batch_bool<float, 4>::batch_bool(bool b)
        : m_value(_mm_castsi128_ps(_mm_set1_epi32(-(int)b)))
    {
    }

    inline batch_bool<float, 4>::batch_bool(bool b0, bool b1, bool b2, bool b3)
        : m_value(_mm_castsi128_ps(_mm_setr_epi32(-(int)b0, -(int)b1, -(int)b2, -(int)b3)))
    {
    }

    inline batch_bool<float, 4>::batch_bool(const __m128& rhs)
        : m_value(rhs)
    {
    }

    inline batch_bool<float, 4>& batch_bool<float, 4>::operator=(const __m128& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch_bool<float, 4>::operator __m128() const
    {
        return m_value;
    }

    inline batch_bool<float, 4> operator&(const batch_bool<float, 4>& lhs, const batch_bool<float, 4>& rhs)
    {
        return _mm_and_ps(lhs, rhs);
    }

    inline batch_bool<float, 4> operator|(const batch_bool<float, 4>& lhs, const batch_bool<float, 4>& rhs)
    {
        return _mm_or_ps(lhs, rhs);
    }

    inline batch_bool<float, 4> operator^(const batch_bool<float, 4>& lhs, const batch_bool<float, 4>& rhs)
    {
        return _mm_xor_ps(lhs, rhs);
    }

    inline batch_bool<float, 4> operator~(const batch_bool<float, 4>& rhs)
    {
        return _mm_xor_ps(rhs, _mm_castsi128_ps(_mm_set1_epi32(-1)));
    }

    inline batch_bool<float, 4> bitwise_andnot(const batch_bool<float, 4>& lhs, const batch_bool<float, 4>& rhs)
    {
        return _mm_andnot_ps(lhs, rhs);
    }

    inline batch_bool<float, 4> operator==(const batch_bool<float, 4>& lhs, const batch_bool<float, 4>& rhs)
    {
        return _mm_cmpeq_ps(lhs, rhs);
    }

    inline batch_bool<float, 4> operator!=(const batch_bool<float, 4>& lhs, const batch_bool<float, 4>& rhs)
    {
        return _mm_cmpneq_ps(lhs, rhs);
    }

    inline bool all(const batch_bool<float, 4>& rhs)
    {
        return _mm_movemask_ps(rhs) == 0x0F;
    }

    inline bool any(const batch_bool<float, 4>& rhs)
    {
        return _mm_movemask_ps(rhs) != 0;
    }

    /**********************************
     * batch<float, 4> implementation *
     **********************************/

    inline batch<float, 4>::batch()
    {
    }

    inline batch<float, 4>::batch(float f)
        : m_value(_mm_set1_ps(f))
    {
    }

    inline batch<float, 4>::batch(float f0, float f1, float f2, float f3)
        : m_value(_mm_setr_ps(f0, f1, f2, f3))
    {
    }

    inline batch<float, 4>::batch(const float* src)
        : m_value(_mm_loadu_ps(src))
    {
    }

    inline batch<float, 4>::batch(const float* src, aligned_mode)
        : m_value(_mm_load_ps(src))
    {
    }

    inline batch<float, 4>::batch(const float* src, unaligned_mode)
        : m_value(_mm_loadu_ps(src))
    {
    }

    inline batch<float, 4>::batch(const __m128& rhs)
        : m_value(rhs)
    {
    }

    inline batch<float, 4>& batch<float, 4>::operator=(const __m128& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<float, 4>::operator __m128() const
    {
        return m_value;
    }

    inline batch<float, 4>& batch<float, 4>::load_aligned(const float* src)
    {
        m_value = _mm_load_ps(src);
        return *this;
    }

    inline batch<float, 4>& batch<float, 4>::load_unaligned(const float* src)
    {
        m_value = _mm_loadu_ps(src);
        return *this;
    }

    inline batch<float, 4>& batch<float, 4>::load_aligned(const double* src)
    {
        __m128 tmp1 = _mm_cvtpd_ps(_mm_load_pd(src));
        __m128 tmp2 = _mm_cvtpd_ps(_mm_load_pd(src+2));
        m_value = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1, 0, 1, 0));
        return *this;
    }

    inline batch<float, 4>& batch<float, 4>::load_unaligned(const double* src)
    {
        __m128 tmp1 = _mm_cvtpd_ps(_mm_loadu_pd(src));
        __m128 tmp2 = _mm_cvtpd_ps(_mm_loadu_pd(src + 2));
        m_value = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1, 0, 1, 0));
        return *this;
    }

    inline batch<float, 4>& batch<float, 4>::load_aligned(const int32_t* src)
    {
        m_value = _mm_cvtepi32_ps(_mm_load_si128((__m128i const*)src));
        return *this;
    }

    inline batch<float, 4>& batch<float, 4>::load_unaligned(const int32_t* src)
    {
        m_value = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i const*)src));
        return *this;
    }

    inline batch<float, 4>& batch<float, 4>::load_aligned(const int64_t* src)
    {
        alignas(16) float tmp[4];
        tmp[0] = float(src[0]);
        tmp[1] = float(src[1]);
        tmp[2] = float(src[2]);
        tmp[3] = float(src[3]);
        m_value = _mm_load_ps(tmp);
        return *this;
    }

    inline batch<float, 4>& batch<float, 4>::load_unaligned(const int64_t* src)
    {
        return load_aligned(src);
    }

    inline void batch<float, 4>::store_aligned(float* dst) const
    {
        _mm_store_ps(dst, m_value);
    }

    inline void batch<float, 4>::store_unaligned(float* dst) const
    {
        _mm_storeu_ps(dst, m_value);
    }

    inline void batch<float, 4>::store_aligned(double* dst) const
    {
        __m128d tmp1 = _mm_cvtps_pd(m_value);
        __m128 ftmp = _mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(3, 2, 3, 2));
        __m128d tmp2 = _mm_cvtps_pd(ftmp);
        _mm_store_pd(dst, tmp1);
        _mm_store_pd(dst + 2, tmp2);
    }

    inline void batch<float, 4>::store_unaligned(double* dst) const
    {
        __m128d tmp1 = _mm_cvtps_pd(m_value);
        __m128 ftmp = _mm_shuffle_ps(m_value, m_value, _MM_SHUFFLE(3, 2, 3, 2));
        __m128d tmp2 = _mm_cvtps_pd(ftmp);
        _mm_storeu_pd(dst, tmp1);
        _mm_storeu_pd(dst + 2, tmp2);
    }

    inline void batch<float, 4>::store_aligned(int32_t* dst) const
    {
        _mm_store_si128((__m128i*)dst, _mm_cvtps_epi32(m_value));
    }

    inline void batch<float, 4>::store_unaligned(int32_t* dst) const
    {
        _mm_storeu_si128((__m128i*)dst, _mm_cvtps_epi32(m_value));
    }

    inline void batch<float, 4>::store_aligned(int64_t* dst) const
    {
        alignas(16) float tmp[4];
        _mm_store_ps(tmp, m_value);
        dst[0] = static_cast<int64_t>(tmp[0]);
        dst[1] = static_cast<int64_t>(tmp[1]);
        dst[2] = static_cast<int64_t>(tmp[2]);
        dst[3] = static_cast<int64_t>(tmp[3]);
    }

    inline void batch<float, 4>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline float batch<float, 4>::operator[](std::size_t index) const
    {
        alignas(16) float x[4];
        store_aligned(x);
        return x[index & 3];
    }

    inline batch<float, 4> operator-(const batch<float, 4>& rhs)
    {
        return _mm_xor_ps(rhs, _mm_castsi128_ps(_mm_set1_epi32(0x80000000)));
    }

    inline batch<float, 4> operator+(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_add_ps(lhs, rhs);
    }

    inline batch<float, 4> operator-(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_sub_ps(lhs, rhs);
    }

    inline batch<float, 4> operator*(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_mul_ps(lhs, rhs);
    }

    inline batch<float, 4> operator/(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_div_ps(lhs, rhs);
    }

    inline batch_bool<float, 4> operator==(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_cmpeq_ps(lhs, rhs);
    }

    inline batch_bool<float, 4> operator!=(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_cmpneq_ps(lhs, rhs);
    }

    inline batch_bool<float, 4> operator<(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_cmplt_ps(lhs, rhs);
    }

    inline batch_bool<float, 4> operator<=(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_cmple_ps(lhs, rhs);
    }

    inline batch<float, 4> operator&(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_and_ps(lhs, rhs);
    }

    inline batch<float, 4> operator|(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_or_ps(lhs, rhs);
    }

    inline batch<float, 4> operator^(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_xor_ps(lhs, rhs);
    }

    inline batch<float, 4> operator~(const batch<float, 4>& rhs)
    {
        return _mm_xor_ps(rhs, _mm_castsi128_ps(_mm_set1_epi32(-1)));
    }

    inline batch<float, 4> bitwise_andnot(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_andnot_ps(lhs, rhs);
    }

    inline batch<float, 4> min(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_min_ps(lhs, rhs);
    }

    inline batch<float, 4> max(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return _mm_max_ps(lhs, rhs);
    }

    inline batch<float, 4> fmin(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return min(lhs, rhs);
    }

    inline batch<float, 4> fmax(const batch<float, 4>& lhs, const batch<float, 4>& rhs)
    {
        return max(lhs, rhs);
    }

    inline batch<float, 4> abs(const batch<float, 4>& rhs)
    {
        __m128 sign_mask = _mm_set1_ps(-0.f);  // -0.f = 1 << 31
        return _mm_andnot_ps(sign_mask, rhs);
    }

    inline batch<float, 4> fabs(const batch<float, 4>& rhs)
    {
        return abs(rhs);
    }

    inline batch<float, 4> sqrt(const batch<float, 4>& rhs)
    {
        return _mm_sqrt_ps(rhs);
    }

    inline batch<float, 4> fma(const batch<float, 4>& x, const batch<float, 4>& y, const batch<float, 4>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm_fmadd_ps(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm_macc_ps(x, y, z);
#else
        return x * y + z;
#endif
    }

    inline batch<float, 4> fms(const batch<float, 4>& x, const batch<float, 4>& y, const batch<float, 4>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm_fmsub_ps(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm_msub_ps(x, y, z);
#else
        return x * y - z;
#endif
    }

    inline batch<float, 4> fnma(const batch<float, 4>& x, const batch<float, 4>& y, const batch<float, 4>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm_fnmadd_ps(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm_nmacc_ps(x, y, z);
#else
        return -x * y + z;
#endif
    }

    inline batch<float, 4> fnms(const batch<float, 4>& x, const batch<float, 4>& y, const batch<float, 4>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm_fnmsub_ps(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm_nmsub_ps(x, y, z);
#else
        return -x * y - z;
#endif
    }

    inline float hadd(const batch<float, 4>& rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE3_VERSION
        __m128 tmp0 = _mm_hadd_ps(rhs, rhs);
        __m128 tmp1 = _mm_hadd_ps(tmp0, tmp0);
#else
        __m128 tmp0 = _mm_add_ps(rhs, _mm_movehl_ps(rhs, rhs));
        __m128 tmp1 = _mm_add_ss(tmp0, _mm_shuffle_ps(tmp0, tmp0, 1));
#endif
        return _mm_cvtss_f32(tmp1);
    }

    inline batch<float, 4> haddp(const batch<float, 4>* row)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE3_VERSION
        return _mm_hadd_ps(_mm_hadd_ps(row[0], row[1]),
                           _mm_hadd_ps(row[2], row[3]));
#else
        __m128 tmp0 = _mm_unpacklo_ps(row[0], row[1]);
        __m128 tmp1 = _mm_unpackhi_ps(row[0], row[1]);
        __m128 tmp2 = _mm_unpackhi_ps(row[2], row[3]);
        tmp0 = _mm_add_ps(tmp0, tmp1);
        tmp1 = _mm_unpacklo_ps(row[2], row[3]);
        tmp1 = _mm_add_ps(tmp1, tmp2);
        tmp2 = _mm_movehl_ps(tmp1, tmp0);
        tmp0 = _mm_movelh_ps(tmp0, tmp1);
        return _mm_add_ps(tmp0, tmp2);
#endif
    }

    inline batch<float, 4> select(const batch_bool<float, 4>& cond, const batch<float, 4>& a, const batch<float, 4>& b)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        return _mm_blendv_ps(b, a, cond);
#else
        return _mm_or_ps(_mm_and_ps(cond, a), _mm_andnot_ps(cond, b));
#endif
    }

    inline batch_bool<float, 4> isnan(const batch<float, 4>& x)
    {
        return _mm_cmpunord_ps(x, x);
    }
}

#endif
