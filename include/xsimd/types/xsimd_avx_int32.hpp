/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX_INT32_HPP
#define XSIMD_AVX_INT32_HPP

#include <cstdint>

#include "xsimd_base.hpp"
#include "xsimd_int_conversion.hpp"

namespace xsimd
{

    /**************************
     * batch_bool<int32_t, 8> *
     **************************/

    template <>
    struct simd_batch_traits<batch_bool<int32_t, 8>>
    {
        using value_type = int32_t;
        static constexpr std::size_t size = 8;
        using batch_type = batch<int32_t, 8>;
        static constexpr std::size_t align = 32;
    };

    template <>
    class batch_bool<int32_t, 8> : public simd_batch_bool<batch_bool<int32_t, 8>>
    {
    public:

        batch_bool();
        explicit batch_bool(bool b);
        batch_bool(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7);
        batch_bool(const __m256i& rhs);
        batch_bool& operator=(const __m256i& rhs);

        operator __m256i() const;

        bool operator[](std::size_t index) const;

    private:

        __m256i m_value;
    };

    /*********************
     * batch<int32_t, 8> *
     *********************/

    template <>
    struct simd_batch_traits<batch<int32_t, 8>>
    {
        using value_type = int32_t;
        static constexpr std::size_t size = 8;
        using batch_bool_type = batch_bool<int32_t, 8>;
        static constexpr std::size_t align = 32;
    };

    template <>
    class batch<int32_t, 8> : public simd_batch<batch<int32_t, 8>>
    {
    public:

        using self_type = batch<int32_t, 8>;
        using base_type = simd_batch<self_type>;

        batch();
        explicit batch(int32_t i);
        batch(int32_t i0, int32_t i1, int32_t i2, int32_t i3, int32_t i4, int32_t i5, int32_t i6, int32_t i7);
        explicit batch(const int32_t* src);
        batch(const int32_t* src, aligned_mode);
        batch(const int32_t* src, unaligned_mode);
        batch(const __m256i& rhs);
        batch& operator=(const __m256i& rhs);

        operator __m256i() const;

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

        __m256i m_value;
    };

    batch<int32_t, 8> operator<<(const batch<int32_t, 8>& lhs, int32_t rhs);
    batch<int32_t, 8> operator>>(const batch<int32_t, 8>& lhs, int32_t rhs);

    /*****************************************
     * batch_bool<int32_t, 8> implementation *
     *****************************************/

#if XSIMD_X86_INSTR_SET < XSIMD_X86_AVX2_VERSION

#define XSIMD_SPLIT_AVX(avx_name)                              \
    __m128i avx_name##_low = _mm256_castsi256_si128(avx_name); \
    __m128i avx_name##_high = _mm256_extractf128_si256(avx_name, 1)

#define XSIMD_RETURN_MERGED_SSE(res_low, res_high)    \
    __m256i result = _mm256_castsi128_si256(res_low); \
    return _mm256_insertf128_si256(result, res_high, 1)

#define XSIMD_APPLY_SSE_FUNCTION(func, avx_lhs, avx_rhs)     \
    XSIMD_SPLIT_AVX(avx_lhs);                                \
    XSIMD_SPLIT_AVX(avx_rhs);                                \
    __m128i res_low = func(avx_lhs##_low, avx_rhs##_low);    \
    __m128i res_high = func(avx_lhs##_high, avx_rhs##_high); \
    XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif

    inline batch_bool<int32_t, 8>::batch_bool()
    {
    }

    inline batch_bool<int32_t, 8>::batch_bool(bool b)
        : m_value(_mm256_set1_epi32(-(int32_t)b))
    {
    }

    inline batch_bool<int32_t, 8>::batch_bool(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7)
        : m_value(_mm256_setr_epi32(-(int32_t)b0, -(int32_t)b1, -(int32_t)b2, -(int32_t)b3, -(int32_t)b4, -(int32_t)b5, -(int32_t)b6, -(int32_t)b7))
    {
    }

    inline batch_bool<int32_t, 8>::batch_bool(const __m256i& rhs)
        : m_value(rhs)
    {
    }

    inline batch_bool<int32_t, 8>& batch_bool<int32_t, 8>::operator=(const __m256i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch_bool<int32_t, 8>::operator __m256i() const
    {
        return m_value;
    }

    inline bool batch_bool<int32_t, 8>::operator[](std::size_t index) const
    {
        alignas(32) int32_t x[8];
        _mm256_store_si256((__m256i*)x, m_value);
        return static_cast<bool>(x[index & 7]);
    }

    namespace detail
    {
        template <>
        struct batch_bool_kernel<int32_t, 8>
        {
            using batch_type = batch_bool<int32_t, 8>;

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_and_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_and_si128, lhs, rhs);
#endif
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_or_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_or_si128, lhs, rhs);
#endif
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_xor_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_xor_si128, lhs, rhs);
#endif
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_xor_si256(rhs, _mm256_set1_epi32(-1));
#else
                XSIMD_SPLIT_AVX(rhs);
                __m128i res_low = _mm_xor_si128(rhs_low, _mm_set1_epi32(-1));
                __m128i res_high = _mm_xor_si128(rhs_high, _mm_set1_epi32(-1));
                XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_andnot_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_andnot_si128, lhs, rhs);
#endif
            }

            static batch_type equal(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_cmpeq_epi32(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_cmpeq_epi32, lhs, rhs);
#endif
            }

            static batch_type not_equal(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(lhs == rhs);
            }

            static bool all(const batch_type& rhs)
            {
                return _mm256_testc_si256(rhs, batch_bool<int32_t, 8>(true)) != 0;
            }

            static bool any(const batch_type& rhs)
            {
                return !_mm256_testz_si256(rhs, rhs);
            }
        };
    }

    /************************************
     * batch<int32_t, 8> implementation *
     ************************************/

    inline batch<int32_t, 8>::batch()
    {
    }

    inline batch<int32_t, 8>::batch(int32_t i)
        : m_value(_mm256_set1_epi32(i))
    {
    }

    inline batch<int32_t, 8>::batch(int32_t i0, int32_t i1, int32_t i2, int32_t i3, int32_t i4, int32_t i5, int32_t i6, int32_t i7)
        : m_value(_mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7))
    {
    }

    inline batch<int32_t, 8>::batch(const int32_t* src)
        : m_value(_mm256_loadu_si256((__m256i const*)src))
    {
    }

    inline batch<int32_t, 8>::batch(const int32_t* src, aligned_mode)
        : m_value(_mm256_load_si256((__m256i const*)src))
    {
    }

    inline batch<int32_t, 8>::batch(const int32_t* src, unaligned_mode)
        : m_value(_mm256_loadu_si256((__m256i const*)src))
    {
    }

    inline batch<int32_t, 8>::batch(const __m256i& rhs)
        : m_value(rhs)
    {
    }

    inline batch<int32_t, 8>& batch<int32_t, 8>::operator=(const __m256i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<int32_t, 8>::operator __m256i() const
    {
        return m_value;
    }

    inline batch<int32_t, 8>& batch<int32_t, 8>::load_aligned(const int32_t* src)
    {
        m_value = _mm256_load_si256((__m256i const*)src);
        return *this;
    }

    inline batch<int32_t, 8>& batch<int32_t, 8>::load_unaligned(const int32_t* src)
    {
        m_value = _mm256_loadu_si256((__m256i const*)src);
        return *this;
    }

    inline batch<int32_t, 8>& batch<int32_t, 8>::load_aligned(const int64_t* src)
    {
        alignas(32) int32_t tmp[8];
        unroller<8>([&](std::size_t i){
            tmp[i] = static_cast<int32_t>(src[i]);
        });
        m_value = _mm256_load_si256((__m256i const*)tmp);
        return *this;
    }

    inline batch<int32_t, 8>& batch<int32_t, 8>::load_unaligned(const int64_t* src)
    {
        return load_aligned(src);
    }

    inline batch<int32_t, 8>& batch<int32_t, 8>::load_aligned(const float* src)
    {
        m_value = _mm256_cvtps_epi32(_mm256_load_ps(src));
        return *this;
    }

    inline batch<int32_t, 8>& batch<int32_t, 8>::load_unaligned(const float* src)
    {
        m_value = _mm256_cvtps_epi32(_mm256_loadu_ps(src));
        return *this;
    }

    inline batch<int32_t, 8>& batch<int32_t, 8>::load_aligned(const double* src)
    {
        __m128i tmp1 = _mm256_cvtpd_epi32(_mm256_load_pd(src));
        __m128i tmp2 = _mm256_cvtpd_epi32(_mm256_load_pd(src + 4));
        m_value = _mm256_castsi128_si256(tmp1);
        m_value = _mm256_insertf128_si256(m_value, tmp2, 1);
        return *this;
    }

    inline batch<int32_t, 8>& batch<int32_t, 8>::load_unaligned(const double* src)
    {
        __m128i tmp1 = _mm256_cvtpd_epi32(_mm256_loadu_pd(src));
        __m128i tmp2 = _mm256_cvtpd_epi32(_mm256_loadu_pd(src + 4));
        m_value = _mm256_castsi128_si256(tmp1);
        m_value = _mm256_insertf128_si256(m_value, tmp2, 1);
        return *this;
    }

    inline batch<int32_t, 8>& batch<int32_t, 8>::load_aligned(const int8_t* src)
    {
        __m128i tmp = _mm_loadl_epi64((const __m128i*)src);
        m_value = detail::xsimd_cvtepi8_epi32(tmp);
        return *this;
    }

    inline batch<int32_t, 8>& batch<int32_t, 8>::load_unaligned(const int8_t* src)
    {
        return load_aligned(src);
    }

    inline batch<int32_t, 8>& batch<int32_t, 8>::load_aligned(const uint8_t* src)
    {
        __m128i tmp = _mm_loadl_epi64((const __m128i*)src);
        m_value = detail::xsimd_cvtepu8_epi32(tmp);
        return *this;
    }

    inline batch<int32_t, 8>& batch<int32_t, 8>::load_unaligned(const uint8_t* src)
    {
        return load_aligned(src);
    }

    inline void batch<int32_t, 8>::store_aligned(int32_t* dst) const
    {
        _mm256_store_si256((__m256i*)dst, m_value);
    }

    inline void batch<int32_t, 8>::store_unaligned(int32_t* dst) const
    {
        _mm256_storeu_si256((__m256i*)dst, m_value);
    }

    inline void batch<int32_t, 8>::store_aligned(int64_t* dst) const
    {
        alignas(32) int32_t tmp[8];
        store_aligned(tmp);
        unroller<8>([&](std::size_t i) {
            dst[i] = static_cast<int64_t>(tmp[i]);
        });
    }

    inline void batch<int32_t, 8>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int32_t, 8>::store_aligned(float* dst) const
    {
        _mm256_store_ps(dst, _mm256_cvtepi32_ps(m_value));
    }

    inline void batch<int32_t, 8>::store_unaligned(float* dst) const
    {
        _mm256_storeu_ps(dst, _mm256_cvtepi32_ps(m_value));
    }

    inline void batch<int32_t, 8>::store_aligned(double* dst) const
    {
        __m128i tmp1 = _mm256_extractf128_si256(m_value, 0);
        __m128i tmp2 = _mm256_extractf128_si256(m_value, 1);
        _mm256_store_pd(dst, _mm256_cvtepi32_pd(tmp1));
        _mm256_store_pd(dst + 4 , _mm256_cvtepi32_pd(tmp2));
    }

    inline void batch<int32_t, 8>::store_unaligned(double* dst) const
    {
        __m128i tmp1 = _mm256_extractf128_si256(m_value, 0);
        __m128i tmp2 = _mm256_extractf128_si256(m_value, 1);
        _mm256_storeu_pd(dst, _mm256_cvtepi32_pd(tmp1));
        _mm256_storeu_pd(dst + 4, _mm256_cvtepi32_pd(tmp2));
    }

    inline void batch<int32_t, 8>::store_aligned(int8_t* dst) const
    {
        __m128i tmp = detail::xsimd_cvtepi32_epi8(m_value);
        _mm_storel_epi64((__m128i*)dst, tmp);
    }

    inline void batch<int32_t, 8>::store_unaligned(int8_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int32_t, 8>::store_aligned(uint8_t* dst) const
    {
        __m128i tmp = detail::xsimd_cvtepi32_epu8(m_value);
        _mm_storel_epi64((__m128i*)dst, tmp);
    }

    inline void batch<int32_t, 8>::store_unaligned(uint8_t* dst) const
    {
        store_aligned(dst);
    }

    inline int32_t batch<int32_t, 8>::operator[](std::size_t index) const
    {
        alignas(32) int32_t x[8];
        store_aligned(x);
        return x[index & 7];
    }

    namespace detail
    {
        template <>
        struct batch_kernel<int32_t, 8>
        {
            using batch_type = batch<int32_t, 8>;
            using value_type = int32_t;
            using batch_bool_type = batch_bool<int32_t, 8>;

            static batch_type neg(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_sub_epi32(_mm256_setzero_si256(), rhs);
#else
                XSIMD_SPLIT_AVX(rhs);
                __m128i res_low = _mm_sub_epi32(_mm_setzero_si128(), rhs_low);
                __m128i res_high = _mm_sub_epi32(_mm_setzero_si128(), rhs_high);
                XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
            }

            static batch_type add(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_add_epi32(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_add_epi32, lhs, rhs);
#endif
            }

            static batch_type sub(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_sub_epi32(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_sub_epi32, lhs, rhs);
#endif
            }

            static batch_type mul(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_mullo_epi32(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_mullo_epi32, lhs, rhs);
#endif
            }

            static batch_type div(const batch_type& lhs, const batch_type& rhs)
            {
#if defined(XSIMD_FAST_INTEGER_DIVISION)
                return _mm256_cvttps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(lhs), _mm256_cvtepi32_ps(rhs)));
#else
                alignas(64) int32_t tmp_lhs[8], tmp_rhs[8], tmp_res[8];
                lhs.store_aligned(tmp_lhs);
                rhs.store_aligned(tmp_rhs);
                unroller<8>([&](std::size_t i) {
                    tmp_res[i] = tmp_lhs[i] / tmp_rhs[i];
                });
                return batch_type(tmp_res, aligned_mode());
#endif
            }

            static batch_type mod(const batch_type& lhs, const batch_type& rhs)
            {
                alignas(64) int32_t tmp_lhs[8], tmp_rhs[8], tmp_res[8];
                lhs.store_aligned(tmp_lhs);
                rhs.store_aligned(tmp_rhs);
                unroller<8>([&](std::size_t i) {
                    tmp_res[i] = tmp_lhs[i] % tmp_rhs[i];
                });
                return batch_type(tmp_res, aligned_mode());
            }

            static batch_bool_type eq(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_cmpeq_epi32(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_cmpeq_epi32, lhs, rhs);
#endif
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(lhs == rhs);
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_cmpgt_epi32(rhs, lhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_cmpgt_epi32, rhs, lhs);
#endif
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(rhs < lhs);
            }

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_and_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_and_si128, lhs, rhs);
#endif
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_or_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_or_si128, lhs, rhs);
#endif
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_xor_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_xor_si128, lhs, rhs);
#endif
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_xor_si256(rhs, _mm256_set1_epi32(-1));
#else
                XSIMD_SPLIT_AVX(rhs);
                __m128i res_low = _mm_xor_si128(rhs_low, _mm_set1_epi32(-1));
                __m128i res_high = _mm_xor_si128(rhs_high, _mm_set1_epi32(-1));
                XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_andnot_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_andnot_si128, lhs, rhs);
#endif
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_min_epi32(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_min_epi32, lhs, rhs);
#endif
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_max_epi32(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_max_epi32, lhs, rhs);
#endif
            }

            static batch_type abs(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_sign_epi32(rhs, rhs);
#else
                XSIMD_SPLIT_AVX(rhs);
                __m128i res_low = _mm_sign_epi32(rhs_low, rhs_low);
                __m128i res_high = _mm_sign_epi32(rhs_high, rhs_high);
                XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
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
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                __m256i tmp1 = _mm256_hadd_epi32(rhs, rhs);
                __m256i tmp2 = _mm256_hadd_epi32(tmp1, tmp1);
                __m128i tmp3 = _mm256_extracti128_si256(tmp2, 1);
                __m128i tmp4 = _mm_add_epi32(_mm256_castsi256_si128(tmp2), tmp3);
                return _mm_cvtsi128_si32(tmp4);
#else
                XSIMD_SPLIT_AVX(rhs);
                __m128i tmp1 = _mm_add_epi32(rhs_low, rhs_high);
                __m128i tmp2 = _mm_hadd_epi32(tmp1, tmp1);
                __m128i tmp3 = _mm_hadd_epi32(tmp2, tmp2);
                return _mm_cvtsi128_si32(tmp3);
#endif
            }

            static batch_type select(const batch_bool_type& cond, const batch_type& a, const batch_type& b)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_blendv_epi8(b, a, cond);
#else
                XSIMD_SPLIT_AVX(cond);
                XSIMD_SPLIT_AVX(a);
                XSIMD_SPLIT_AVX(b);
                __m128i res_low = _mm_blendv_epi8(b_low, a_low, cond_low);
                __m128i res_high = _mm_blendv_epi8(b_high, a_high, cond_high);
                XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
            }
        };
    }

    inline batch<int32_t, 8> operator<<(const batch<int32_t, 8>& lhs, int32_t rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
        return _mm256_slli_epi32(lhs, rhs);
#else
        XSIMD_SPLIT_AVX(lhs);
        __m128i res_low = _mm_slli_epi32(lhs_low, rhs);
        __m128i res_high = _mm_slli_epi32(lhs_high, rhs);
        XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
    }

    inline batch<int32_t, 8> operator>>(const batch<int32_t, 8>& lhs, int32_t rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
        return _mm256_srli_epi32(lhs, rhs);
#else
        XSIMD_SPLIT_AVX(lhs);
        __m128i res_low = _mm_srli_epi32(lhs_low, rhs);
        __m128i res_high = _mm_srli_epi32(lhs_high, rhs);
        XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
    }
}

#undef XSIMD_APPLY_SSE_FUNCTION
#undef XSIMD_RETURN_MERGED_SSE
#undef XSIMD_SPLIT_AVX

#endif
