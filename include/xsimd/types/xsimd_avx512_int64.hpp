
/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_INT64_HPP
#define XSIMD_AVX512_INT64_HPP

#include <cstdint>

#include "xsimd_base.hpp"

namespace xsimd
{

    /**************************
     * batch_bool<int64_t, 8> *
     **************************/

    template <>
    struct simd_batch_traits<batch_bool<int64_t, 8>>
    {
        using value_type = int64_t;
        static constexpr std::size_t size = 8;
        using batch_type = batch<int64_t, 8>;
        static constexpr std::size_t align = 0;
    };

    template <>
    class batch_bool<int64_t, 8> : 
        public batch_bool_avx512<__mmask8, batch_bool<int64_t, 8>>,
        public simd_batch_bool<batch_bool<int64_t, 8>>
    {
    public:
        using base_class = batch_bool_avx512<__mmask8, batch_bool<int64_t, 8>>;
        using base_class::base_class;
    };

    namespace detail
    {
        template <>
        struct batch_bool_kernel<int64_t, 8>
            : batch_bool_kernel_avx512<int64_t, 8>
        {
        };
    }

    /*********************
     * batch<int64_t, 8> *
     *********************/

    template <>
    struct simd_batch_traits<batch<int64_t, 8>>
    {
        using value_type = int64_t;
        static constexpr std::size_t size = 8;
        using batch_bool_type = batch_bool<int64_t, 8>;
        static constexpr std::size_t align = 64;
    };

    template <>
    class batch<int64_t, 8> : public simd_batch<batch<int64_t, 8>>
    {
    public:

        using self_type = batch<int64_t, 8>;
        using base_type = simd_batch<self_type>;

        batch();
        explicit batch(int64_t i);
        batch(int64_t i0, int64_t i1,  int64_t i2,  int64_t i3,  int64_t i4,  int64_t i5,  int64_t i6,  int64_t i7);
        explicit batch(const int64_t* src);
        batch(const int64_t* src, aligned_mode);
        batch(const int64_t* src, unaligned_mode);
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

        int64_t operator[](std::size_t index) const;

    private:

        __m512i m_value;
    };

    batch<int64_t, 8> operator<<(const batch<int64_t, 8>& lhs, int32_t rhs);
    batch<int64_t, 8> operator>>(const batch<int64_t, 8>& lhs, int32_t rhs);
    batch<int64_t, 8> operator<<(const batch<int64_t, 8>& lhs, const batch<int64_t, 8>& rhs);
    batch<int64_t, 8> operator>>(const batch<int64_t, 8>& lhs, const batch<int64_t, 8>& rhs);

    /************************************
     * batch<int64_t, 8> implementation *
     ************************************/

    inline batch<int64_t, 8>::batch()
    {
    }

    inline batch<int64_t, 8>::batch(int64_t i)
        : m_value(_mm512_set1_epi64(i))
    {
    }

    inline batch<int64_t, 8>::batch(int64_t i0, int64_t i1,  int64_t i2,  int64_t i3,  int64_t i4,  int64_t i5,  int64_t i6,  int64_t i7)
        : m_value(_mm512_setr_epi64(i0, i1, i2, i3, i4, i5, i6, i7))
    {
    }

    inline batch<int64_t, 8>::batch(const int64_t* src)
        : m_value(_mm512_loadu_si512(src))
    {
    }

    inline batch<int64_t, 8>::batch(const int64_t* src, aligned_mode)
        : m_value(_mm512_load_epi64(src))
    {
    }

    inline batch<int64_t, 8>::batch(const int64_t* src, unaligned_mode)
        : m_value(_mm512_loadu_si512(src))
    {
    }

    inline batch<int64_t, 8>::batch(const __m512i& rhs)
        : m_value(rhs)
    {
    }

    inline batch<int64_t, 8>& batch<int64_t, 8>::operator=(const __m512i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<int64_t, 8>::operator __m512i() const
    {
        return m_value;
    }

    inline batch<int64_t, 8>& batch<int64_t, 8>::load_aligned(const int32_t* src)
    {
        m_value = _mm512_cvtepi32_epi64(_mm256_load_si256((const __m256i *) src));
        return *this;
    }

    inline batch<int64_t, 8>& batch<int64_t, 8>::load_unaligned(const int32_t* src)
    {
        m_value = _mm512_cvtepi32_epi64(_mm256_loadu_si256((const __m256i *) src));
        return *this;
    }

    inline batch<int64_t, 8>& batch<int64_t, 8>::load_aligned(const int64_t* src)
    {
        m_value = _mm512_load_epi64(src);
        return *this;
    }

    inline batch<int64_t, 8>& batch<int64_t, 8>::load_unaligned(const int64_t* src)
    {
        m_value = _mm512_loadu_si512(src);
        return *this;
    }

    inline batch<int64_t, 8>& batch<int64_t, 8>::load_aligned(const float* src)
    {
        m_value = _mm512_cvtps_epi64(_mm256_load_ps(src));
        return *this;
    }

    inline batch<int64_t, 8>& batch<int64_t, 8>::load_unaligned(const float* src)
    {
        m_value = _mm512_cvtps_epi64(_mm256_loadu_ps(src));
        return *this;
    }

    inline batch<int64_t, 8>& batch<int64_t, 8>::load_aligned(const double* src)
    {
        m_value = _mm512_cvttpd_epi64(_mm512_load_pd(src));
        return *this;
    }

    inline batch<int64_t, 8>& batch<int64_t, 8>::load_unaligned(const double* src)
    {
        m_value = _mm512_cvttpd_epi64(_mm512_loadu_pd(src));
        return *this;
    }

    inline batch<int64_t, 8>& batch<int64_t, 8>::load_aligned(const int8_t* src)
    {
        __m128i tmp = _mm_loadl_epi64((const __m128i*)src);
        m_value = _mm512_cvtepi8_epi64(tmp);
        return *this;
    }

    inline batch<int64_t, 8>& batch<int64_t, 8>::load_unaligned(const int8_t* src)
    {
        return load_aligned(src);
    }

    inline batch<int64_t, 8>& batch<int64_t, 8>::load_aligned(const uint8_t* src)
    {
        __m128i tmp = _mm_loadl_epi64((const __m128i*)src);
        m_value = _mm512_cvtepu8_epi64(tmp);
        return *this;
    }

    inline batch<int64_t, 8>& batch<int64_t, 8>::load_unaligned(const uint8_t* src)
    {
        return load_aligned(src);
    }

    inline void batch<int64_t, 8>::store_aligned(int32_t* dst) const
    {
        _mm256_store_si256((__m256i*)dst, _mm512_cvtepi64_epi32(m_value));
    }

    inline void batch<int64_t, 8>::store_unaligned(int32_t* dst) const
    {
        _mm256_store_si256((__m256i*)dst, _mm512_cvtepi64_epi32(m_value));
    }

    inline void batch<int64_t, 8>::store_aligned(int64_t* dst) const
    {
        _mm512_store_si512(dst, m_value);
    }

    inline void batch<int64_t, 8>::store_unaligned(int64_t* dst) const
    {
        // TODO check if differnt instructions available
        _mm512_store_si512(dst, m_value);
    }

    inline void batch<int64_t, 8>::store_aligned(float* dst) const
    {
        _mm256_store_ps(dst, _mm512_cvtepi64_ps(m_value));
    }

    inline void batch<int64_t, 8>::store_unaligned(float* dst) const
    {
        _mm256_storeu_ps(dst, _mm512_cvtepi64_ps(m_value));
    }

    inline void batch<int64_t, 8>::store_aligned(double* dst) const
    {
        _mm512_store_pd(dst, _mm512_cvtepi64_pd(m_value));
    }

    inline void batch<int64_t, 8>::store_unaligned(double* dst) const
    {
        _mm512_storeu_pd(dst, _mm512_cvtepi64_pd(m_value));
    }

    inline void batch<int64_t, 8>::store_aligned(int8_t* dst) const
    {
        __m128i tmp = _mm512_cvtepi64_epi8(m_value);
        _mm_storel_epi64((__m128i*)dst, tmp);
    }

    inline void batch<int64_t, 8>::store_unaligned(int8_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int64_t, 8>::store_aligned(uint8_t* dst) const
    {
        __m128i tmp = _mm512_cvtusepi64_epi8(m_value);
        _mm_storel_epi64((__m128i*)dst, tmp);
    }

    inline void batch<int64_t, 8>::store_unaligned(uint8_t* dst) const
    {
        store_aligned(dst);
    }

    inline int64_t batch<int64_t, 8>::operator[](std::size_t index) const
    {
        alignas(64) int64_t x[8];
        store_aligned(x);
        return x[index & 7];
    }

    namespace detail
    {
        template <>
        struct batch_kernel<int64_t, 8>
        {
            using batch_type = batch<int64_t, 8>;
            using value_type = int64_t;
            using batch_bool_type = batch_bool<int64_t, 8>;

            static batch_type neg(const batch_type& rhs)
            {
                return _mm512_sub_epi64(_mm512_setzero_si512(), rhs);
            }

            static batch_type add(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_add_epi64(lhs, rhs);
            }

            static batch_type sub(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_sub_epi64(lhs, rhs);
            }

            static batch_type mul(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_mullo_epi64(lhs, rhs);
            }

            static batch_type div(const batch_type& lhs, const batch_type& rhs)
            {
#if defined(XSIMD_FAST_INTEGER_DIVISION)
                return _mm512_cvttpd_epi64(_mm512_div_pd(_mm512_cvtepi64_pd(lhs), _mm512_cvtepi64_pd(rhs)));
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
                return _mm512_cmp_epi64_mask(lhs, rhs, _MM_CMPINT_EQ);
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmp_epi64_mask(lhs, rhs, _MM_CMPINT_NE);
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmp_epi64_mask(lhs, rhs, _MM_CMPINT_LT);
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmp_epi64_mask(lhs, rhs, _MM_CMPINT_LE);
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
                return _mm512_xor_si512(rhs, _mm512_set1_epi64(-1));
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_andnot_si512(lhs, rhs);
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_min_epi64(lhs, rhs);
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_max_epi64(lhs, rhs);
            }

            static batch_type abs(const batch_type& rhs)
            {
                return _mm512_abs_epi64(rhs);
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
                __m256i tmp1 = _mm512_extracti32x8_epi32(rhs, 0);
                __m256i tmp2 = _mm512_extracti32x8_epi32(rhs, 1);
                __m256i res1 = tmp1 + tmp2;
                return xsimd::hadd(batch<int64_t, 4>(res1));
            }

            static batch_type select(const batch_bool_type& cond, const batch_type& a, const batch_type& b)
            {
                return _mm512_mask_blend_epi64(cond, b, a);
            }
        };
    }

    inline batch<int64_t, 8> operator<<(const batch<int64_t, 8>& lhs, int32_t rhs)
    {
        return _mm512_slli_epi64(lhs, rhs);
    }

    inline batch<int64_t, 8> operator>>(const batch<int64_t, 8>& lhs, int32_t rhs)
    {
        return _mm512_srli_epi64(lhs, rhs);
    }

    inline batch<int64_t, 8> operator<<(const batch<int64_t, 8>& lhs, const batch<int64_t, 8>& rhs)
    {
        return _mm512_sllv_epi64(lhs, rhs);
    }

    inline batch<int64_t, 8> operator>>(const batch<int64_t, 8>& lhs, const batch<int64_t, 8>& rhs)
    {
        return _mm512_srlv_epi64(lhs, rhs);
    }
}

#endif
