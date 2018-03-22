/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_SSE_INT32_HPP
#define XSIMD_SSE_INT32_HPP

#include <cstdint>

#include "xsimd_base.hpp"

namespace xsimd
{

    /**************************
     * batch_bool<int32_t, 4> *
     **************************/

    template <>
    struct simd_batch_traits<batch_bool<int32_t, 4>>
    {
        using value_type = bool;
        static constexpr std::size_t size = 4;
        using batch_type = batch<int32_t, 4>;
        static constexpr std::size_t align = 16;
    };

    template <>
    class batch_bool<int32_t, 4> : public simd_batch_bool<batch_bool<int32_t, 4>>
    {
    public:

        batch_bool();
        explicit batch_bool(bool b);
        batch_bool(bool b0, bool b1, bool b2, bool b3);
        batch_bool(const __m128i& rhs);
        batch_bool& operator=(const __m128i& rhs);

        operator __m128i() const;

    private:

        __m128i m_value;
    };

    batch_bool<int32_t, 4> operator&(const batch_bool<int32_t, 4>& lhs, const batch_bool<int32_t, 4>& rhs);
    batch_bool<int32_t, 4> operator|(const batch_bool<int32_t, 4>& lhs, const batch_bool<int32_t, 4>& rhs);
    batch_bool<int32_t, 4> operator^(const batch_bool<int32_t, 4>& lhs, const batch_bool<int32_t, 4>& rhs);
    batch_bool<int32_t, 4> operator~(const batch_bool<int32_t, 4>& rhs);
    batch_bool<int32_t, 4> bitwise_andnot(const batch_bool<int32_t, 4>& lhs, const batch_bool<int32_t, 4>& rhs);

    batch_bool<int32_t, 4> operator==(const batch_bool<int32_t, 4>& lhs, const batch_bool<int32_t, 4>& rhs);
    batch_bool<int32_t, 4> operator!=(const batch_bool<int32_t, 4>& lhs, const batch_bool<int32_t, 4>& rhs);

    bool all(const batch_bool<int32_t, 4>& rhs);
    bool any(const batch_bool<int32_t, 4>& rhs);

    /*********************
     * batch<int32_t, 4> *
     *********************/

    template <>
    struct simd_batch_traits<batch<int32_t, 4>>
    {
        using value_type = int32_t;
        static constexpr std::size_t size = 4;
        using batch_bool_type = batch_bool<int32_t, 4>;
        static constexpr std::size_t align = 16;
    };

    template <>
    class batch<int32_t, 4> : public simd_batch<batch<int32_t, 4>>
    {
    public:

        batch();
        explicit batch(int32_t i);
        batch(int32_t i0, int32_t i1, int32_t i2, int32_t i3);
        explicit batch(const int32_t* src);
        batch(const int32_t* src, aligned_mode);
        batch(const int32_t* src, unaligned_mode);
        batch(const __m128i& rhs);
        batch& operator=(const __m128i& rhs);

        operator __m128i() const;

        batch& load_aligned(const int32_t* src);
        batch& load_unaligned(const int32_t* src);

        batch& load_aligned(const int64_t* src);
        batch& load_unaligned(const int64_t* src);

        batch& load_aligned(const float* src);
        batch& load_unaligned(const float* src);

        batch& load_aligned(const double* src);
        batch& load_unaligned(const double* src);

        void store_aligned(int32_t* dst) const;
        void store_unaligned(int32_t* dst) const;

        void store_aligned(int64_t* dst) const;
        void store_unaligned(int64_t* dst) const;

        void store_aligned(float* dst) const;
        void store_unaligned(float* dst) const;

        void store_aligned(double* dst) const;
        void store_unaligned(double* dst) const;

        int32_t operator[](std::size_t index) const;

    private:

        __m128i m_value;
    };

    batch<int32_t, 4> operator-(const batch<int32_t, 4>& rhs);
    batch<int32_t, 4> operator+(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);
    batch<int32_t, 4> operator-(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);
    batch<int32_t, 4> operator*(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);
    batch<int32_t, 4> operator/(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);

    batch_bool<int32_t, 4> operator==(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);
    batch_bool<int32_t, 4> operator!=(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);
    batch_bool<int32_t, 4> operator<(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);
    batch_bool<int32_t, 4> operator<=(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);

    batch<int32_t, 4> operator&(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);
    batch<int32_t, 4> operator|(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);
    batch<int32_t, 4> operator^(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);
    batch<int32_t, 4> operator~(const batch<int32_t, 4>& rhs);
    batch<int32_t, 4> bitwise_andnot(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);

    batch<int32_t, 4> min(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);
    batch<int32_t, 4> max(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs);

    batch<int32_t, 4> abs(const batch<int32_t, 4>& rhs);

    batch<int32_t, 4> fma(const batch<int32_t, 4>& x, const batch<int32_t, 4>& y, const batch<int32_t, 4>& z);
    batch<int32_t, 4> fms(const batch<int32_t, 4>& x, const batch<int32_t, 4>& y, const batch<int32_t, 4>& z);
    batch<int32_t, 4> fnma(const batch<int32_t, 4>& x, const batch<int32_t, 4>& y, const batch<int32_t, 4>& z);
    batch<int32_t, 4> fnms(const batch<int32_t, 4>& x, const batch<int32_t, 4>& y, const batch<int32_t, 4>& z);

    int32_t hadd(const batch<int32_t, 4>& rhs);

    batch<int32_t, 4> select(const batch_bool<int32_t, 4>& cond, const batch<int32_t, 4>& a, const batch<int32_t, 4>& b);

    batch<int32_t, 4> operator<<(const batch<int32_t, 4>& lhs, int32_t rhs);
    batch<int32_t, 4> operator>>(const batch<int32_t, 4>& lhs, int32_t rhs);

    /*****************************************
     * batch_bool<int32_t, 4> implementation *
     *****************************************/

    inline batch_bool<int32_t, 4>::batch_bool()
    {
    }

    inline batch_bool<int32_t, 4>::batch_bool(bool b)
        : m_value(_mm_set1_epi32(-(int32_t)b))
    {
    }

    inline batch_bool<int32_t, 4>::batch_bool(bool b0, bool b1, bool b2, bool b3)
        : m_value(_mm_setr_epi32(-(int32_t)b0, -(int32_t)b1, -(int32_t)b2, -(int32_t)b3))
    {
    }

    inline batch_bool<int32_t, 4>::batch_bool(const __m128i& rhs)
        : m_value(rhs)
    {
    }

    inline batch_bool<int32_t, 4>& batch_bool<int32_t, 4>::operator=(const __m128i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch_bool<int32_t, 4>::operator __m128i() const
    {
        return m_value;
    }

    inline batch_bool<int32_t, 4> operator&(const batch_bool<int32_t, 4>& lhs, const batch_bool<int32_t, 4>& rhs)
    {
        return _mm_and_si128(lhs, rhs);
    }

    inline batch_bool<int32_t, 4> operator|(const batch_bool<int32_t, 4>& lhs, const batch_bool<int32_t, 4>& rhs)
    {
        return _mm_or_si128(lhs, rhs);
    }

    inline batch_bool<int32_t, 4> operator^(const batch_bool<int32_t, 4>& lhs, const batch_bool<int32_t, 4>& rhs)
    {
        return _mm_xor_si128(lhs, rhs);
    }

    inline batch_bool<int32_t, 4> operator~(const batch_bool<int32_t, 4>& rhs)
    {
        return _mm_xor_si128(rhs, _mm_set1_epi32(-1));
    }

    inline batch_bool<int32_t, 4> bitwise_andnot(const batch_bool<int32_t, 4>& lhs, const batch_bool<int32_t, 4>& rhs)
    {
        return _mm_andnot_si128(lhs, rhs);
    }

    inline batch_bool<int32_t, 4> operator==(const batch_bool<int32_t, 4>& lhs, const batch_bool<int32_t, 4>& rhs)
    {
        return _mm_cmpeq_epi32(lhs, rhs);
    }

    inline batch_bool<int32_t, 4> operator!=(const batch_bool<int32_t, 4>& lhs, const batch_bool<int32_t, 4>& rhs)
    {
        return ~(lhs == rhs);
    }

    inline bool all(const batch_bool<int32_t, 4>& rhs)
    {
        return _mm_movemask_epi8(rhs) == 0xFFFF;
    }

    inline bool any(const batch_bool<int32_t, 4>& rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        return !_mm_testz_si128(rhs, rhs);
#else
        return _mm_movemask_epi8(rhs) != 0;
#endif
    }

    /************************************
     * batch<int32_t, 4> implementation *
     ************************************/

    inline batch<int32_t, 4>::batch()
    {
    }

    inline batch<int32_t, 4>::batch(int32_t i)
        : m_value(_mm_set1_epi32(i))
    {
    }

    inline batch<int32_t, 4>::batch(int32_t i0, int32_t i1, int32_t i2, int32_t i3)
        : m_value(_mm_setr_epi32(i0, i1, i2, i3))
    {
    }

    inline batch<int32_t, 4>::batch(const int32_t* src)
        : m_value(_mm_loadu_si128((__m128i const*)src))
    {
    }

    inline batch<int32_t, 4>::batch(const int32_t* src, aligned_mode)
        : m_value(_mm_load_si128((__m128i const*)src))
    {
    }

    inline batch<int32_t, 4>::batch(const int32_t* src, unaligned_mode)
        : m_value(_mm_loadu_si128((__m128i const*)src))
    {
    }

    inline batch<int32_t, 4>::batch(const __m128i& rhs)
        : m_value(rhs)
    {
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::operator=(const __m128i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<int32_t, 4>::operator __m128i() const
    {
        return m_value;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_aligned(const int32_t* src)
    {
        m_value = _mm_load_si128((__m128i const*)src);
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_unaligned(const int32_t* src)
    {
        m_value = _mm_loadu_si128((__m128i const*)src);
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_aligned(const int64_t* src)
    {
        alignas(16) int32_t tmp[4];
        tmp[0] = static_cast<int32_t>(src[0]);
        tmp[1] = static_cast<int32_t>(src[1]);
        tmp[2] = static_cast<int32_t>(src[2]);
        tmp[3] = static_cast<int32_t>(src[3]);
        return load_aligned(tmp);
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_unaligned(const int64_t* src)
    {
        return load_aligned(src);
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_aligned(const float* src)
    {
        m_value = _mm_cvtps_epi32(_mm_load_ps(src));
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_unaligned(const float* src)
    {
        m_value = _mm_cvtps_epi32(_mm_loadu_ps(src));
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_aligned(const double* src)
    {
        __m128i tmp1 = _mm_cvtpd_epi32(_mm_load_pd(src));
        __m128i tmp2 = _mm_cvtpd_epi32(_mm_load_pd(src + 2));
        m_value = _mm_unpacklo_epi64(tmp1, tmp2);
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_unaligned(const double* src)
    {
        __m128i tmp1 = _mm_cvtpd_epi32(_mm_loadu_pd(src));
        __m128i tmp2 = _mm_cvtpd_epi32(_mm_loadu_pd(src + 2));
        m_value = _mm_unpacklo_epi64(tmp1, tmp2);
        return *this;
    }

    inline void batch<int32_t, 4>::store_aligned(int32_t* dst) const
    {
        _mm_store_si128((__m128i*)dst, m_value);
    }

    inline void batch<int32_t, 4>::store_unaligned(int32_t* dst) const
    {
        _mm_storeu_si128((__m128i*)dst, m_value);
    }

    inline void batch<int32_t, 4>::store_aligned(int64_t* dst) const
    {
        alignas(16) int32_t tmp[4];
        store_aligned(tmp);
        dst[0] = int64_t(tmp[0]);
        dst[1] = int64_t(tmp[1]);
        dst[2] = int64_t(tmp[2]);
        dst[3] = int64_t(tmp[3]);
    }

    inline void batch<int32_t, 4>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int32_t, 4>::store_aligned(float* dst) const
    {
        _mm_store_ps(dst, _mm_cvtepi32_ps(m_value));
    }

    inline void batch<int32_t, 4>::store_unaligned(float* dst) const
    {
        _mm_storeu_ps(dst, _mm_cvtepi32_ps(m_value));
    }

    inline void batch<int32_t, 4>::store_aligned(double* dst) const
    {
        __m128d tmp1 = _mm_cvtepi32_pd(m_value);
        __m128d tmp2 = _mm_cvtepi32_pd(_mm_unpackhi_epi64(m_value, m_value));
        _mm_store_pd(dst, tmp1);
        _mm_store_pd(dst + 2, tmp2);
    }

    inline void batch<int32_t, 4>::store_unaligned(double* dst) const
    {
        __m128d tmp1 = _mm_cvtepi32_pd(m_value);
        __m128d tmp2 = _mm_cvtepi32_pd(_mm_unpackhi_epi64(m_value, m_value));
        _mm_storeu_pd(dst, tmp1);
        _mm_storeu_pd(dst + 2, tmp2);
    }

    inline int32_t batch<int32_t, 4>::operator[](std::size_t index) const
    {
        alignas(16) int32_t x[4];
        store_aligned(x);
        return x[index & 3];
    }

    inline batch<int32_t, 4> operator-(const batch<int32_t, 4>& rhs)
    {
        return _mm_sub_epi32(_mm_setzero_si128(), rhs);
    }

    inline batch<int32_t, 4> operator+(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return _mm_add_epi32(lhs, rhs);
    }

    inline batch<int32_t, 4> operator-(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return _mm_sub_epi32(lhs, rhs);
    }

    inline batch<int32_t, 4> operator*(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        return _mm_mullo_epi32(lhs, rhs);
#else
        __m128i a13 = _mm_shuffle_epi32(lhs, 0xF5);
        __m128i b13 = _mm_shuffle_epi32(rhs, 0xF5);
        __m128i prod02 = _mm_mul_epu32(lhs, rhs);
        __m128i prod13 = _mm_mul_epu32(a13, b13);
        __m128i prod01 = _mm_unpacklo_epi32(prod02, prod13);
        __m128i prod23 = _mm_unpackhi_epi32(prod02, prod13);
        return _mm_unpacklo_epi64(prod01, prod23);
#endif
    }

    inline batch<int32_t, 4> operator/(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(lhs), _mm_cvtepi32_ps(rhs)));
    }

    inline batch_bool<int32_t, 4> operator==(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return _mm_cmpeq_epi32(lhs, rhs);
    }

    inline batch_bool<int32_t, 4> operator!=(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return ~(lhs == rhs);
    }

    inline batch_bool<int32_t, 4> operator<(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return _mm_cmplt_epi32(lhs, rhs);
    }

    inline batch_bool<int32_t, 4> operator<=(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return ~(rhs < lhs);
    }

    inline batch<int32_t, 4> operator&(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return _mm_and_si128(lhs, rhs);
    }

    inline batch<int32_t, 4> operator|(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return _mm_or_si128(lhs, rhs);
    }

    inline batch<int32_t, 4> operator^(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return _mm_xor_si128(lhs, rhs);
    }

    inline batch<int32_t, 4> operator~(const batch<int32_t, 4>& rhs)
    {
        return _mm_xor_si128(rhs, _mm_set1_epi32(-1));
    }

    inline batch<int32_t, 4> bitwise_andnot(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
        return _mm_andnot_si128(lhs, rhs);
    }

    inline batch<int32_t, 4> min(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        return _mm_min_epi32(lhs, rhs);
#else
        __m128i greater = _mm_cmpgt_epi32(lhs, rhs);
        return select(greater, rhs, lhs);
#endif
    }

    inline batch<int32_t, 4> max(const batch<int32_t, 4>& lhs, const batch<int32_t, 4>& rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        return _mm_max_epi32(lhs, rhs);
#else
        __m128i greater = _mm_cmpgt_epi32(lhs, rhs);
        return select(greater, lhs, rhs);
#endif
    }

    inline batch<int32_t, 4> abs(const batch<int32_t, 4>& rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSSE3_VERSION
        return _mm_sign_epi32(rhs, rhs);
#else
        __m128i sign = _mm_srai_epi32(rhs, 31);
        __m128i inv = _mm_xor_si128(rhs, sign);
        return _mm_sub_epi32(inv, sign);
#endif
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
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSSE3_VERSION
        __m128i tmp1 = _mm_hadd_epi32(rhs, rhs);
        __m128i tmp2 = _mm_hadd_epi32(tmp1, tmp1);
        return _mm_cvtsi128_si32(tmp2);
#else
        __m128i tmp1 = _mm_shuffle_epi32(rhs, 0x0E);
        __m128i tmp2 = _mm_add_epi32(rhs, tmp1);
        __m128i tmp3 = _mm_shuffle_epi32(tmp2, 0x01);
        __m128i tmp4 = _mm_add_epi32(tmp2, tmp3);
        return _mm_cvtsi128_si32(tmp4);
#endif
    }

    inline batch<int32_t, 4> select(const batch_bool<int32_t, 4>& cond, const batch<int32_t, 4>& a, const batch<int32_t, 4>& b)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        return _mm_blendv_epi8(b, a, cond);
#else
        return _mm_or_si128(_mm_and_si128(cond, a), _mm_andnot_si128(cond, b));
#endif
    }

    inline batch<int32_t, 4> operator<<(const batch<int32_t, 4>& lhs, int32_t rhs)
    {
        return _mm_slli_epi32(lhs, rhs);
    }

    inline batch<int32_t, 4> operator>>(const batch<int32_t, 4>& lhs, int32_t rhs)
    {
        return _mm_srli_epi32(lhs, rhs);
    }
}

#endif
