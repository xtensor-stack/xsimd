/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_INT32_HPP
#define XSIMD_AVX512_INT32_HPP

#include <cstdint>

#include "xsimd_base.hpp"

namespace xsimd
{

    /**************************
     * batch_bool<int32_t, 16> *
     **************************/

    template <>
    struct simd_batch_traits<batch_bool<int32_t, 16>>
    {
        using value_type = int32_t;
        static constexpr std::size_t size = 16;
        using batch_type = batch<int32_t, 16>;
        static constexpr std::size_t align = 0;
    };

    template <>
    class batch_bool<int32_t, 16> : 
        public batch_bool_avx512<__mmask16, batch_bool<int32_t, 16>>,
        public simd_batch_bool<batch_bool<int32_t, 16>>
    {
    public:
        using base_class = batch_bool_avx512<__mmask16, batch_bool<int32_t, 16>>;
        using base_class::base_class;
    };

    namespace detail
    {
        template <>
        struct batch_bool_kernel<int32_t, 16>
            : batch_bool_kernel_avx512<int32_t, 16>
        {
        };
    }

    /*********************
     * batch<int32_t, 16> *
     *********************/

    template <>
    struct simd_batch_traits<batch<int32_t, 16>>
    {
        using value_type = int32_t;
        static constexpr std::size_t size = 16;
        using batch_bool_type = batch_bool<int32_t, 16>;
        static constexpr std::size_t align = 64;
    };

    template <>
    class batch<int32_t, 16> : public simd_batch<batch<int32_t, 16>>
    {
    public:

        using self_type = batch<int32_t, 16>;
        using base_type = simd_batch<self_type>;

        batch();
        explicit batch(int32_t i);
        batch(int32_t i0, int32_t i1,  int32_t i2,  int32_t i3,  int32_t i4,  int32_t i5,  int32_t i6,  int32_t i7,
              int32_t i8, int32_t i9, int32_t i10, int32_t i11, int32_t i12, int32_t i13, int32_t i14, int32_t i15);
        explicit batch(const int32_t* src);
        batch(const int32_t* src, aligned_mode);
        batch(const int32_t* src, unaligned_mode);
        batch(const __m512i& rhs);
        batch& operator=(const __m512i& rhs);

        operator __m512i() const;

        batch& load_aligned(const int32_t* src);
        batch& load_unaligned(const int32_t* src);

        batch& load_aligned(const int64_t* src);
        batch& load_unaligned(const int64_t* src);

        batch& load_aligned(const float* src);
        batch& load_unaligned(const float* src);

        batch& load_aligned(const double* src);
        batch& load_unaligned(const double* src);

        batch& load_aligned(const int8_t* src);
        batch& load_unaligned(const int8_t* src);

        batch& load_aligned(const uint8_t* src);
        batch& load_unaligned(const uint8_t* src);

        void store_aligned(int32_t* dst) const;
        void store_unaligned(int32_t* dst) const;

        void store_aligned(int64_t* dst) const;
        void store_unaligned(int64_t* dst) const;

        void store_aligned(float* dst) const;
        void store_unaligned(float* dst) const;

        void store_aligned(double* dst) const;
        void store_unaligned(double* dst) const;

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

        __m512i m_value;
    };

    batch<int32_t, 16> operator<<(const batch<int32_t, 16>& lhs, int32_t rhs);
    batch<int32_t, 16> operator>>(const batch<int32_t, 16>& lhs, int32_t rhs);
    batch<int32_t, 16> operator<<(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch<int32_t, 16> operator>>(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);

    /************************************
     * batch<int32_t, 16> implementation *
     ************************************/

    inline batch<int32_t, 16>::batch()
    {
    }

    inline batch<int32_t, 16>::batch(int32_t i)
        : m_value(_mm512_set1_epi32(i))
    {
    }

    inline batch<int32_t, 16>::batch(int32_t i0, int32_t i1,  int32_t i2,  int32_t i3,  int32_t i4,  int32_t i5,  int32_t i6,  int32_t i7,
                                     int32_t i8, int32_t i9, int32_t i10, int32_t i11, int32_t i12, int32_t i13, int32_t i14, int32_t i15)
        : m_value(_mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15))
    {
    }

    inline batch<int32_t, 16>::batch(const int32_t* src)
        : m_value(_mm512_loadu_si512((__m512i const*)src))
    {
    }

    inline batch<int32_t, 16>::batch(const int32_t* src, aligned_mode)
        : m_value(_mm512_load_si512((__m512i const*)src))
    {
    }

    inline batch<int32_t, 16>::batch(const int32_t* src, unaligned_mode)
        : m_value(_mm512_loadu_si512((__m512i const*)src))
    {
    }

    inline batch<int32_t, 16>::batch(const __m512i& rhs)
        : m_value(rhs)
    {
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::operator=(const __m512i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<int32_t, 16>::operator __m512i() const
    {
        return m_value;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const int32_t* src)
    {
        m_value = _mm512_load_si512((__m512i const*)src);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const int32_t* src)
    {
        m_value = _mm512_loadu_si512((__m512i const*)src);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const int64_t* src)
    {
        alignas(64) int32_t tmp[16];
        tmp[0] = static_cast<int32_t>(src[0]);
        tmp[1] = static_cast<int32_t>(src[1]);
        tmp[2] = static_cast<int32_t>(src[2]);
        tmp[3] = static_cast<int32_t>(src[3]);
        tmp[4] = static_cast<int32_t>(src[4]);
        tmp[5] = static_cast<int32_t>(src[5]);
        tmp[6] = static_cast<int32_t>(src[6]);
        tmp[7] = static_cast<int32_t>(src[7]);
        tmp[8] = static_cast<int32_t>(src[8]);
        tmp[9] = static_cast<int32_t>(src[9]);
        tmp[10] = static_cast<int32_t>(src[10]);
        tmp[11] = static_cast<int32_t>(src[11]);
        tmp[12] = static_cast<int32_t>(src[12]);
        tmp[13] = static_cast<int32_t>(src[13]);
        tmp[14] = static_cast<int32_t>(src[14]);
        tmp[15] = static_cast<int32_t>(src[15]);
        m_value = _mm512_load_si512((__m512i const*)tmp);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const int64_t* src)
    {
        return load_aligned(src);
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const float* src)
    {
        m_value = _mm512_cvtps_epi32(_mm512_load_ps(src));
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const float* src)
    {
        m_value = _mm512_cvtps_epi32(_mm512_loadu_ps(src));
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const double* src)
    {
        __m256i tmp1 = _mm512_cvtpd_epi32(_mm512_load_pd(src));
        __m256i tmp2 = _mm512_cvtpd_epi32(_mm512_load_pd(src + 8));
        m_value = _mm512_castsi256_si512(tmp1);
        m_value = _mm512_inserti32x8(m_value, tmp2, 1);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const double* src)
    {
        __m256i tmp1 = _mm512_cvtpd_epi32(_mm512_loadu_pd(src));
        __m256i tmp2 = _mm512_cvtpd_epi32(_mm512_loadu_pd(src + 8));
        m_value = _mm512_castsi256_si512(tmp1);
        m_value = _mm512_inserti32x8(m_value, tmp2, 1);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const int8_t* src)
    {
        __m128i tmp = _mm_load_si128((const __m128i*)src);
        m_value = _mm512_cvtepi8_epi32(tmp);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const int8_t* src)
    {
        __m128i tmp = _mm_loadu_si128((const __m128i*)src);
        m_value = _mm512_cvtepi8_epi32(tmp);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const uint8_t* src)
    {
        __m128i tmp = _mm_load_si128((const __m128i*)src);
        m_value = _mm512_cvtepu8_epi32(tmp);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const uint8_t* src)
    {
        __m128i tmp = _mm_loadu_si128((const __m128i*)src);
        m_value = _mm512_cvtepu8_epi32(tmp);
        return *this;
    }

    inline void batch<int32_t, 16>::store_aligned(int32_t* dst) const
    {
        _mm512_store_si512((__m512i*)dst, m_value);
    }

    inline void batch<int32_t, 16>::store_unaligned(int32_t* dst) const
    {
        _mm512_storeu_si512((__m512i*)dst, m_value);
    }

    inline void batch<int32_t, 16>::store_aligned(int64_t* dst) const
    {
        alignas(64) int32_t tmp[16];
        store_aligned(tmp);
        dst[0] = int64_t(tmp[0]);
        dst[1] = int64_t(tmp[1]);
        dst[2] = int64_t(tmp[2]);
        dst[3] = int64_t(tmp[3]);
        dst[4] = int64_t(tmp[4]);
        dst[5] = int64_t(tmp[5]);
        dst[6] = int64_t(tmp[6]);
        dst[7] = int64_t(tmp[7]);
        dst[8] = int64_t(tmp[8]);
        dst[9] = int64_t(tmp[9]);
        dst[10] = int64_t(tmp[10]);
        dst[11] = int64_t(tmp[11]);
        dst[12] = int64_t(tmp[12]);
        dst[13] = int64_t(tmp[13]);
        dst[14] = int64_t(tmp[14]);
        dst[15] = int64_t(tmp[15]);
    }

    inline void batch<int32_t, 16>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int32_t, 16>::store_aligned(float* dst) const
    {
        _mm512_store_ps(dst, _mm512_cvtepi32_ps(m_value));
    }

    inline void batch<int32_t, 16>::store_unaligned(float* dst) const
    {
        _mm512_storeu_ps(dst, _mm512_cvtepi32_ps(m_value));
    }

    inline void batch<int32_t, 16>::store_aligned(double* dst) const
    {
        __m256i tmp1 = _mm512_extracti32x8_epi32(m_value, 0);
        __m256i tmp2 = _mm512_extracti32x8_epi32(m_value, 1);
        _mm512_store_pd(dst, _mm512_cvtepi32_pd(tmp1));
        _mm512_store_pd(dst + 8 , _mm512_cvtepi32_pd(tmp2));
    }

    inline void batch<int32_t, 16>::store_unaligned(double* dst) const
    {
        __m256i tmp1 = _mm512_extracti32x8_epi32(m_value, 0);
        __m256i tmp2 = _mm512_extracti32x8_epi32(m_value, 1);
        _mm512_store_pd(dst, _mm512_cvtepi32_pd(tmp1));
        _mm512_store_pd(dst + 8 , _mm512_cvtepi32_pd(tmp2));
    }

    inline void batch<int32_t, 16>::store_aligned(int8_t* dst) const
    {
        __m128i tmp = _mm512_cvtepi32_epi8(m_value);
        _mm_store_si128((__m128i*)dst, tmp);
    }

    inline void batch<int32_t, 16>::store_unaligned(int8_t* dst) const
    {
        __m128i tmp = _mm512_cvtepi32_epi8(m_value);
        _mm_storeu_si128((__m128i*)dst, tmp);
    }

    inline void batch<int32_t, 16>::store_aligned(uint8_t* dst) const
    {
        __m128i tmp = _mm512_cvtusepi32_epi8(m_value);
        _mm_store_si128((__m128i*)dst, tmp);
    }

    inline void batch<int32_t, 16>::store_unaligned(uint8_t* dst) const
    {
        __m128i tmp = _mm512_cvtusepi32_epi8(m_value);
        _mm_storeu_si128((__m128i*)dst, tmp);
    }

    inline int32_t batch<int32_t, 16>::operator[](std::size_t index) const
    {
        alignas(64) int32_t x[16];
        store_aligned(x);
        return x[index & 15];
    }

    namespace detail
    {
        template <>
        struct batch_kernel<int32_t, 16>
        {
            using batch_type = batch<int32_t, 16>;
            using value_type = int32_t;
            using batch_bool_type = batch_bool<int32_t, 16>;

            static batch_type neg(const batch_type& rhs)
            {
                return _mm512_sub_epi32(_mm512_setzero_si512(), rhs);
            }

            static batch_type add(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_add_epi32(lhs, rhs);
            }

            static batch_type sub(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_sub_epi32(lhs, rhs);
            }

            static batch_type mul(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_mullo_epi32(lhs, rhs);
            }

            static batch_type div(const batch_type& lhs, const batch_type& rhs)
            {
#if defined(XSIMD_FAST_INTEGER_DIVISION)
                return _mm512_cvttps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(lhs), _mm512_cvtepi32_ps(rhs)));
#else
                XSIMD_MACRO_UNROLL_BINARY(/);
#endif
            }

            static batch_type mod(const batch_type& lhs, const batch_type& rhs)
            {
                XSIMD_MACRO_UNROLL_BINARY(%);
            }

            static batch_bool_type eq(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmp_epi32_mask(lhs, rhs, _MM_CMPINT_EQ);
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmp_epi32_mask(lhs, rhs, _MM_CMPINT_NE);
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmp_epi32_mask(lhs, rhs, _MM_CMPINT_LT);
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmp_epi32_mask(lhs, rhs, _MM_CMPINT_LE);
            }

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_and_si512(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_or_si512(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_xor_si512(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return _mm512_xor_si512(rhs, _mm512_set1_epi32(-1));
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_andnot_si512(lhs, rhs);
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_min_epi32(lhs, rhs);
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_max_epi32(lhs, rhs);
            }

            static batch_type abs(const batch_type& rhs)
            {
                return _mm512_abs_epi32(rhs);
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
                // TODO Why not _mm512_reduce_add_...?
                __m256i tmp1 = _mm512_extracti32x8_epi32(rhs, 0);
                __m256i tmp2 = _mm512_extracti32x8_epi32(rhs, 1);
                __m256i res1 = tmp1 + tmp2;
                return xsimd::hadd(batch<int32_t, 8>(res1));
            }

            static batch_type select(const batch_bool_type& cond, const batch_type& a, const batch_type& b)
            {
                return _mm512_mask_blend_epi32(cond, b, a);
            }
        };
    }

    inline batch<int32_t, 16> operator<<(const batch<int32_t, 16>& lhs, int32_t rhs)
    {
        return _mm512_slli_epi32(lhs, rhs);
    }

    inline batch<int32_t, 16> operator>>(const batch<int32_t, 16>& lhs, int32_t rhs)
    {
        return _mm512_srli_epi32(lhs, rhs);
    }

    inline batch<int32_t, 16> operator<<(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_sllv_epi32(lhs, rhs);
    }

    inline batch<int32_t, 16> operator>>(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs)
    {
        return _mm512_srlv_epi32(lhs, rhs);
    }
}

#endif
