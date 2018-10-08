/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_INT32_HPP
#define XSIMD_AVX512_INT32_HPP

#include "xsimd_avx512_bool.hpp"
#include "xsimd_avx512_int_base.hpp"

namespace xsimd
{

    /***************************
     * batch_bool<int32_t, 16> *
     ***************************/

    template <>
    struct simd_batch_traits<batch_bool<int32_t, 16>>
    {
        using value_type = int32_t;
        static constexpr std::size_t size = 16;
        using batch_type = batch<int32_t, 16>;
        static constexpr std::size_t align = 0;
    };

    template <>
    struct simd_batch_traits<batch_bool<uint32_t, 16>>
    {
        using value_type = uint32_t;
        static constexpr std::size_t size = 16;
        using batch_type = batch<uint32_t, 16>;
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

    template <>
    class batch_bool<uint32_t, 16> :
        public batch_bool_avx512<__mmask16, batch_bool<uint32_t, 16>>,
        public simd_batch_bool<batch_bool<uint32_t, 16>>
    {
    public:
        using base_class = batch_bool_avx512<__mmask16, batch_bool<uint32_t, 16>>;
        using base_class::base_class;
    };

    namespace detail
    {
        template <>
        struct batch_bool_kernel<int32_t, 16>
            : batch_bool_kernel_avx512<int32_t, 16>
        {
        };

        template <>
        struct batch_bool_kernel<uint32_t, 16>
            : batch_bool_kernel_avx512<uint32_t, 16>
        {
        };
    }

    /**********************
     * batch<int32_t, 16> *
     **********************/

    template <>
    struct simd_batch_traits<batch<int32_t, 16>>
    {
        using value_type = int32_t;
        static constexpr std::size_t size = 16;
        using batch_bool_type = batch_bool<int32_t, 16>;
        static constexpr std::size_t align = 64;
    };

    template <>
    struct simd_batch_traits<batch<uint32_t, 16>>
    {
        using value_type = uint32_t;
        static constexpr std::size_t size = 16;
        using batch_bool_type = batch_bool<uint32_t, 16>;
        static constexpr std::size_t align = 64;
    };

    template <>
    class batch<int32_t, 16> : public avx512_int_batch<int32_t, 16>
    {
    public:

        using base_type = avx512_int_batch<int32_t, 16>;
        using base_type::base_type;
        using base_type::load_aligned;
        using base_type::load_unaligned;
        using base_type::store_aligned;
        using base_type::store_unaligned;

        XSIMD_DECLARE_LOAD_STORE_INT32(int32_t, 16);
    };

    template <>
    class batch<uint32_t, 16> : public avx512_int_batch<uint32_t, 16>
    {
    public:

        using base_type = avx512_int_batch<uint32_t, 16>;
        using base_type::base_type;
        using base_type::load_aligned;
        using base_type::load_unaligned;
        using base_type::store_aligned;
        using base_type::store_unaligned;

        XSIMD_DECLARE_LOAD_STORE_INT32(uint32_t, 16);
    };

    batch<int32_t, 16> operator<<(const batch<int32_t, 16>& lhs, int32_t rhs);
    batch<int32_t, 16> operator>>(const batch<int32_t, 16>& lhs, int32_t rhs);
    batch<int32_t, 16> operator<<(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch<int32_t, 16> operator>>(const batch<int32_t, 16>& lhs, const batch<int32_t, 16>& rhs);
    batch<uint32_t, 16> operator<<(const batch<uint32_t, 16>& lhs, uint32_t rhs);
    batch<uint32_t, 16> operator>>(const batch<uint32_t, 16>& lhs, uint32_t rhs);
    batch<uint32_t, 16> operator<<(const batch<uint32_t, 16>& lhs, const batch<uint32_t, 16>& rhs);
    batch<uint32_t, 16> operator>>(const batch<uint32_t, 16>& lhs, const batch<uint32_t, 16>& rhs);

    /*************************************
     * batch<int32_t, 16> implementation *
     *************************************/

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const int8_t* src)
    {
        __m128i tmp = _mm_load_si128((const __m128i*)src);
        this->m_value = _mm512_cvtepi8_epi32(tmp);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const int8_t* src)
    {
        __m128i tmp = _mm_loadu_si128((const __m128i*)src);
        this->m_value = _mm512_cvtepi8_epi32(tmp);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const uint8_t* src)
    {
        __m128i tmp = _mm_load_si128((const __m128i*)src);
        this->m_value = _mm512_cvtepu8_epi32(tmp);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const uint8_t* src)
    {
        __m128i tmp = _mm_loadu_si128((const __m128i*)src);
        this->m_value = _mm512_cvtepu8_epi32(tmp);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const int16_t* src)
    {
        __m256i tmp = _mm256_load_si256((const __m256i*)src);
        this->m_value = _mm512_cvtepi16_epi32(tmp);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const int16_t* src)
    {
        __m256i tmp = _mm256_loadu_si256((const __m256i*)src);
        this->m_value = _mm512_cvtepi16_epi32(tmp);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const uint16_t* src)
    {
        __m256i tmp = _mm256_load_si256((const __m256i*)src);
        this->m_value = _mm512_cvtepu16_epi32(tmp);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const uint16_t* src)
    {
        __m256i tmp = _mm256_loadu_si256((const __m256i*)src);
        this->m_value = _mm512_cvtepu16_epi32(tmp);
        return *this;
    }

    //XSIMD_DEFINE_LOAD_STORE(int32_t, 16, uint32_t, 64)
    XSIMD_DEFINE_LOAD_STORE(int32_t, 16, int64_t, 64)
    XSIMD_DEFINE_LOAD_STORE(int32_t, 16, uint64_t, 64)

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const float* src)
    {
        this->m_value = _mm512_cvtps_epi32(_mm512_load_ps(src));
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const float* src)
    {
        this->m_value = _mm512_cvtps_epi32(_mm512_loadu_ps(src));
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_aligned(const double* src)
    {
        __m256i tmp1 = _mm512_cvtpd_epi32(_mm512_load_pd(src));
        __m256i tmp2 = _mm512_cvtpd_epi32(_mm512_load_pd(src + 8));
        this->m_value = _mm512_castsi256_si512(tmp1);
        this->m_value = _mm512_inserti32x8(this->m_value, tmp2, 1);
        return *this;
    }

    inline batch<int32_t, 16>& batch<int32_t, 16>::load_unaligned(const double* src)
    {
        __m256i tmp1 = _mm512_cvtpd_epi32(_mm512_loadu_pd(src));
        __m256i tmp2 = _mm512_cvtpd_epi32(_mm512_loadu_pd(src + 8));
        this->m_value = _mm512_castsi256_si512(tmp1);
        this->m_value = _mm512_inserti32x8(this->m_value, tmp2, 1);
        return *this;
    }

    inline void batch<int32_t, 16>::store_aligned(int8_t* dst) const
    {
        __m128i tmp = _mm512_cvtepi32_epi8(this->m_value);
        _mm_store_si128((__m128i*)dst, tmp);
    }

    inline void batch<int32_t, 16>::store_unaligned(int8_t* dst) const
    {
        __m128i tmp = _mm512_cvtepi32_epi8(this->m_value);
        _mm_storeu_si128((__m128i*)dst, tmp);
    }

    inline void batch<int32_t, 16>::store_aligned(uint8_t* dst) const
    {
        __m128i tmp = _mm512_cvtusepi32_epi8(this->m_value);
        _mm_store_si128((__m128i*)dst, tmp);
    }

    inline void batch<int32_t, 16>::store_unaligned(uint8_t* dst) const
    {
        __m128i tmp = _mm512_cvtusepi32_epi8(this->m_value);
        _mm_storeu_si128((__m128i*)dst, tmp);
    }

    inline void batch<int32_t, 16>::store_aligned(int16_t* dst) const
    {
        __m256i tmp = _mm512_cvtepi32_epi16(this->m_value);
        _mm256_store_si256((__m256i*)dst, tmp);
    }

    inline void batch<int32_t, 16>::store_unaligned(int16_t* dst) const
    {
        __m256i tmp = _mm512_cvtepi32_epi16(this->m_value);
        _mm256_storeu_si256((__m256i*)dst, tmp);
    }

    inline void batch<int32_t, 16>::store_aligned(uint16_t* dst) const
    {
        __m256i tmp = _mm512_cvtusepi32_epi16(this->m_value);
        _mm256_store_si256((__m256i*)dst, tmp);
    }

    inline void batch<int32_t, 16>::store_unaligned(uint16_t* dst) const
    {
        __m256i tmp = _mm512_cvtusepi32_epi16(this->m_value);
        _mm256_storeu_si256((__m256i*)dst, tmp);
    }

    inline void batch<int32_t, 16>::store_aligned(float* dst) const
    {
        _mm512_store_ps(dst, _mm512_cvtepi32_ps(this->m_value));
    }

    inline void batch<int32_t, 16>::store_unaligned(float* dst) const
    {
        _mm512_storeu_ps(dst, _mm512_cvtepi32_ps(this->m_value));
    }

    inline void batch<int32_t, 16>::store_aligned(double* dst) const
    {
        __m256i tmp1 = _mm512_extracti32x8_epi32(this->m_value, 0);
        __m256i tmp2 = _mm512_extracti32x8_epi32(this->m_value, 1);
        _mm512_store_pd(dst, _mm512_cvtepi32_pd(tmp1));
        _mm512_store_pd(dst + 8 , _mm512_cvtepi32_pd(tmp2));
    }

    inline void batch<int32_t, 16>::store_unaligned(double* dst) const
    {
        __m256i tmp1 = _mm512_extracti32x8_epi32(this->m_value, 0);
        __m256i tmp2 = _mm512_extracti32x8_epi32(this->m_value, 1);
        _mm512_store_pd(dst, _mm512_cvtepi32_pd(tmp1));
        _mm512_store_pd(dst + 8 , _mm512_cvtepi32_pd(tmp2));
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
                return _mm512_cmpeq_epi32_mask(lhs, rhs);
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmpneq_epi32_mask(lhs, rhs);
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmplt_epi32_mask(lhs, rhs);
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmple_epi32_mask(lhs, rhs);
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
