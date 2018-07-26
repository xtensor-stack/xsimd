/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX_INT64_HPP
#define XSIMD_AVX_INT64_HPP

#include <cstdint>

#include "xsimd_base.hpp"

namespace xsimd
{

    /**************************
     * batch_bool<int64_t, 4> *
     **************************/

    template <>
    struct simd_batch_traits<batch_bool<int64_t, 4>>
    {
        using value_type = int64_t;
        static constexpr std::size_t size = 4;
        using batch_type = batch<int64_t, 4>;
        static constexpr std::size_t align = 32;
    };

    template <>
    class batch_bool<int64_t, 4> : public simd_batch_bool<batch_bool<int64_t, 4>>
    {
    public:

        batch_bool();
        explicit batch_bool(bool b);
        batch_bool(bool b0, bool b1, bool b2, bool b3);
        batch_bool(const __m256i& rhs);
        batch_bool& operator=(const __m256i& rhs);

        operator __m256i() const;

        bool operator[](std::size_t index) const;

    private:

        __m256i m_value;
    };

    /*********************
     * batch<int64_t, 4> *
     *********************/

    template <>
    struct simd_batch_traits<batch<int64_t, 4>>
    {
        using value_type = int64_t;
        static constexpr std::size_t size = 4;
        using batch_bool_type = batch_bool<int64_t, 4>;
        static constexpr std::size_t align = 32;
    };

    template <>
    class batch<int64_t, 4> : public simd_batch<batch<int64_t, 4>>
    {

    public:

        using self_type = batch<int64_t, 4>;
        using base_type = simd_batch<self_type>;

        batch();
        explicit batch(int64_t i);
        batch(int64_t i0, int64_t i1, int64_t i2, int64_t i3);
        explicit batch(const int64_t* src);
        batch(const int64_t* src, aligned_mode);
        batch(const int64_t* src, unaligned_mode);
        batch(const __m256i& rhs);
        batch& operator=(const __m256i& rhs);

        operator __m256i() const;

        batch& load_aligned(const int64_t* src);
        batch& load_unaligned(const int64_t* src);

        batch& load_aligned(const int32_t* src);
        batch& load_unaligned(const int32_t* src);

        batch& load_aligned(const float* src);
        batch& load_unaligned(const float* src);

        batch& load_aligned(const double* src);
        batch& load_unaligned(const double* src);

        batch& load_aligned(const int8_t* src);
        batch& load_unaligned(const int8_t* src);

        batch& load_aligned(const uint8_t* src);
        batch& load_unaligned(const uint8_t* src);

        void store_aligned(int64_t* dst) const;
        void store_unaligned(int64_t* dst) const;

        void store_aligned(int32_t* dst) const;
        void store_unaligned(int32_t* dst) const;

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

        __m256i m_value;
    };

    batch<int64_t, 4> operator<<(const batch<int64_t, 4>& lhs, int32_t rhs);
    batch<int64_t, 4> operator>>(const batch<int64_t, 4>& lhs, int32_t rhs);

    /*****************************************
     * batch_bool<int64_t, 4> implementation *
     *****************************************/

#if XSIMD_X86_INSTR_SET < XSIMD_X86_AVX2_VERSION

#define XSIMD_SPLIT_AVX(name)                          \
    __m128i name##_low = _mm256_castsi256_si128(name); \
    __m128i name##_high = _mm256_extractf128_si256(name, 1)

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

    inline batch_bool<int64_t, 4>::batch_bool()
    {
    }

    inline batch_bool<int64_t, 4>::batch_bool(bool b)
        : m_value(_mm256_set1_epi64x(-(int64_t)b))
    {
    }

    inline batch_bool<int64_t, 4>::batch_bool(bool b0, bool b1, bool b2, bool b3)
        : m_value(_mm256_setr_epi64x(-(int64_t)b0, -(int64_t)b1, -(int64_t)b2, -(int64_t)b3))
    {
    }

    inline batch_bool<int64_t, 4>::batch_bool(const __m256i& rhs)
        : m_value(rhs)
    {
    }

    inline batch_bool<int64_t, 4>& batch_bool<int64_t, 4>::operator=(const __m256i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch_bool<int64_t, 4>::operator __m256i() const
    {
        return m_value;
    }

    inline bool batch_bool<int64_t, 4>::operator[](std::size_t index) const
    {
        alignas(32) int64_t x[4];
        _mm256_store_si256((__m256i*)x, m_value);
        return static_cast<bool>(x[index & 3]);
    }

    namespace detail
    {
        template <>
        struct batch_bool_kernel<int64_t, 4>
        {
            using batch_type = batch_bool<int64_t, 4>;

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
                return _mm256_cmpeq_epi64(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_cmpeq_epi64, lhs, rhs);
#endif
            }

            static batch_type not_equal(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(lhs == rhs);
            }

            static bool all(const batch_type& rhs)
            {
                return _mm256_testc_si256(rhs, batch_bool<int64_t, 4>(true)) != 0;
            }

            static bool any(const batch_type& rhs)
            {
                return !_mm256_testz_si256(rhs, rhs);
            }
        };
    }

    /************************************
     * batch<int64_t, 4> implementation *
     ************************************/

    inline batch<int64_t, 4>::batch()
    {
    }

    inline batch<int64_t, 4>::batch(int64_t i)
        : m_value(_mm256_set1_epi64x(i))
    {
    }

    inline batch<int64_t, 4>::batch(int64_t i0, int64_t i1, int64_t i2, int64_t i3)
        : m_value(_mm256_setr_epi64x(i0, i1, i2, i3))
    {
    }

    inline batch<int64_t, 4>::batch(const int64_t* src)
        : m_value(_mm256_loadu_si256((__m256i const*)src))
    {
    }

    inline batch<int64_t, 4>::batch(const int64_t* src, aligned_mode)
        : m_value(_mm256_load_si256((__m256i const*)src))
    {
    }

    inline batch<int64_t, 4>::batch(const int64_t* src, unaligned_mode)
        : m_value(_mm256_loadu_si256((__m256i const*)src))
    {
    }

    inline batch<int64_t, 4>::batch(const __m256i& rhs)
        : m_value(rhs)
    {
    }

    inline batch<int64_t, 4>& batch<int64_t, 4>::operator=(const __m256i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<int64_t, 4>::operator __m256i() const
    {
        return m_value;
    }

    inline batch<int64_t, 4>& batch<int64_t, 4>::load_aligned(const int64_t* src)
    {
        m_value = _mm256_load_si256((__m256i const*)src);
        return *this;
    }

    inline batch<int64_t, 4>& batch<int64_t, 4>::load_unaligned(const int64_t* src)
    {
        m_value = _mm256_loadu_si256((__m256i const*)src);
        return *this;
    }

    inline batch<int64_t, 4>& batch<int64_t, 4>::load_aligned(const int32_t* src)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
        m_value = _mm256_cvtepi32_epi64(_mm_load_si128((__m128i const*)src));
        return *this;
#else
        alignas(32) int64_t tmp[4];
        tmp[0] = int64_t(src[0]);
        tmp[1] = int64_t(src[1]);
        tmp[2] = int64_t(src[2]);
        tmp[3] = int64_t(src[3]);
        return load_aligned(tmp);
#endif
    }

    inline batch<int64_t, 4>& batch<int64_t, 4>::load_unaligned(const int32_t* src)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
        m_value = _mm256_cvtepi32_epi64(_mm_loadu_si128((__m128i const*)src));
        return *this;
#else
        return load_aligned(src);
#endif
    }

    inline batch<int64_t, 4>& batch<int64_t, 4>::load_aligned(const float* src)
    {
        alignas(32) int64_t tmp[4];
        tmp[0] = static_cast<int64_t>(src[0]);
        tmp[1] = static_cast<int64_t>(src[1]);
        tmp[2] = static_cast<int64_t>(src[2]);
        tmp[3] = static_cast<int64_t>(src[3]);
        return load_aligned(tmp);
    }

    inline batch<int64_t, 4>& batch<int64_t, 4>::load_unaligned(const float* src)
    {
        return load_aligned(src);
    }

    inline batch<int64_t, 4>& batch<int64_t, 4>::load_aligned(const double* src)
    {
        alignas(32) int64_t tmp[4];
        tmp[0] = static_cast<int64_t>(src[0]);
        tmp[1] = static_cast<int64_t>(src[1]);
        tmp[2] = static_cast<int64_t>(src[2]);
        tmp[3] = static_cast<int64_t>(src[3]);
        return load_aligned(tmp);
    }

    inline batch<int64_t, 4>& batch<int64_t, 4>::load_unaligned(const double* src)
    {
        return load_aligned(src);
    }

    inline batch<int64_t, 4>& batch<int64_t, 4>::load_aligned(const int8_t* src)
    {
        __m128i tmp = _mm_loadl_epi64((const __m128i*)src);
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
        m_value = _mm256_cvtepi8_epi64(tmp);
#else
        __m128i tmp2 = _mm_shufflelo_epi16(tmp, _MM_SHUFFLE(3, 2, 0, 1));
        __m128i tmp_lo = _mm_cvtepi8_epi64(tmp);
        __m128i tmp_hi = _mm_cvtepi8_epi64(tmp2);
        __m256i res = _mm256_castsi128_si256(tmp_lo);
        m_value = _mm256_insertf128_si256(res, tmp_hi, 1);
#endif
        return *this;
    }

    inline batch<int64_t, 4>& batch<int64_t, 4>::load_unaligned(const int8_t* src)
    {
        return load_aligned(src);
    }

    inline batch<int64_t, 4>& batch<int64_t, 4>::load_aligned(const uint8_t* src)
    {
        __m128i tmp = _mm_loadl_epi64((const __m128i*)src);
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
        m_value = _mm256_cvtepu8_epi64(tmp);
#else
        __m128i tmp2 = _mm_shufflelo_epi16(tmp, _MM_SHUFFLE(3, 2, 0, 1));
        __m128i tmp_lo = _mm_cvtepu8_epi64(tmp);
        __m128i tmp_hi = _mm_cvtepu8_epi64(tmp2);
        __m256i res = _mm256_castsi128_si256(tmp_lo);
        m_value = _mm256_insertf128_si256(res, tmp_hi, 1);
#endif
        return *this;
    }

    inline batch<int64_t, 4>& batch<int64_t, 4>::load_unaligned(const uint8_t* src)
    {
        return load_aligned(src);
    }

    inline void batch<int64_t, 4>::store_aligned(int64_t* dst) const
    {
        _mm256_store_si256((__m256i*)dst, m_value);
    }

    inline void batch<int64_t, 4>::store_unaligned(int64_t* dst) const
    {
        _mm256_storeu_si256((__m256i*)dst, m_value);
    }

    inline void batch<int64_t, 4>::store_aligned(int32_t* dst) const
    {
        alignas(32) int64_t tmp[4];
        store_aligned(tmp);
        dst[0] = static_cast<int32_t>(tmp[0]);
        dst[1] = static_cast<int32_t>(tmp[1]);
        dst[2] = static_cast<int32_t>(tmp[2]);
        dst[3] = static_cast<int32_t>(tmp[3]);
    }

    inline void batch<int64_t, 4>::store_unaligned(int32_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int64_t, 4>::store_aligned(float* dst) const
    {
        alignas(32) int64_t tmp[4];
        store_aligned(tmp);
        dst[0] = float(tmp[0]);
        dst[1] = float(tmp[1]);
        dst[2] = float(tmp[2]);
        dst[3] = float(tmp[3]);
    }

    inline void batch<int64_t, 4>::store_unaligned(float* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int64_t, 4>::store_aligned(double* dst) const
    {
        alignas(32) int64_t tmp[4];
        store_aligned(tmp);
        dst[0] = double(tmp[0]);
        dst[1] = double(tmp[1]);
        dst[2] = double(tmp[2]);
        dst[3] = double(tmp[3]);
    }

    inline void batch<int64_t, 4>::store_unaligned(double* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int64_t, 4>::store_aligned(int8_t* dst) const
    {
        alignas(32) int64_t tmp[4];
        store_aligned(tmp);
        dst[0] = static_cast<char>(tmp[0]);
        dst[1] = static_cast<char>(tmp[1]);
        dst[2] = static_cast<char>(tmp[2]);
        dst[3] = static_cast<char>(tmp[3]);
    }

    inline void batch<int64_t, 4>::store_unaligned(int8_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int64_t, 4>::store_aligned(uint8_t* dst) const
    {
        alignas(32) int64_t tmp[4];
        store_aligned(tmp);
        dst[0] = static_cast<unsigned char>(tmp[0]);
        dst[1] = static_cast<unsigned char>(tmp[1]);
        dst[2] = static_cast<unsigned char>(tmp[2]);
        dst[3] = static_cast<unsigned char>(tmp[3]);
    }

    inline void batch<int64_t, 4>::store_unaligned(uint8_t* dst) const
    {
        store_aligned(dst);
    }

    inline int64_t batch<int64_t, 4>::operator[](std::size_t index) const
    {
        alignas(32) int64_t x[4];
        store_aligned(x);
        return x[index & 3];
    }

    namespace detail
    {
        template <>
        struct batch_kernel<int64_t, 4>
        {
            using batch_type = batch<int64_t, 4>;
            using value_type = int64_t;
            using batch_bool_type = batch_bool<int64_t, 4>;

            static batch_type neg(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_sub_epi64(_mm256_setzero_si256(), rhs);
#else
                XSIMD_SPLIT_AVX(rhs);
                __m128i res_low = _mm_sub_epi64(_mm_setzero_si128(), rhs_low);
                __m128i res_high = _mm_sub_epi64(_mm_setzero_si128(), rhs_high);
                XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
            }

            static batch_type add(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_add_epi64(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_add_epi64, lhs, rhs);
#endif
            }

            static batch_type sub(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_sub_epi64(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_sub_epi64, lhs, rhs);
#endif
            }

            static batch_type mul(const batch_type& lhs, const batch_type& rhs)
            {
                alignas(32) int64_t slhs[4], srhs[4];
                lhs.store_aligned(slhs);
                rhs.store_aligned(srhs);
                return batch<int64_t, 4>(slhs[0] * srhs[0], slhs[1] * srhs[1], slhs[2] * srhs[2], slhs[3] * srhs[3]);
            }

            static batch_type div(const batch_type& lhs, const batch_type& rhs)
            {
#if defined(XSIMD_FAST_INTEGER_DIVISION)
                __m256d dlhs = _mm256_setr_pd(static_cast<double>(lhs[0]), static_cast<double>(lhs[1]),
                                              static_cast<double>(lhs[2]), static_cast<double>(lhs[3]));
                __m256d drhs = _mm256_setr_pd(static_cast<double>(rhs[0]), static_cast<double>(rhs[1]),
                                              static_cast<double>(rhs[2]), static_cast<double>(rhs[3]));
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(_mm256_div_pd(dlhs, drhs)));
#else
                using batch_int = batch<int32_t, 4>;
                __m128i tmp = _mm256_cvttpd_epi32(_mm256_div_pd(dlhs, drhs));
                __m128i res_low = _mm_unpacklo_epi32(tmp, batch_int(tmp) < batch_int(0));
                __m128i res_high = _mm_unpackhi_epi32(tmp, batch_int(tmp) < batch_int(0));
                __m256i result = _mm256_castsi128_si256(res_low);
                return _mm256_insertf128_si256(result, res_high, 1);
#endif
#else
                XSIMD_MACRO_UNROLL_BINARY(/)
#endif
            }

            static batch_bool_type eq(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_cmpeq_epi64(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_cmpeq_epi64, lhs, rhs);
#endif
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(lhs == rhs);
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_cmpgt_epi64(rhs, lhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_cmpgt_epi64, rhs, lhs);
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
                return select(lhs < rhs, lhs, rhs);
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
                return select(lhs > rhs, lhs, rhs);
            }

            static batch_type abs(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                __m256i sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), rhs);
                __m256i inv = _mm256_xor_si256(rhs, sign);
                return _mm256_sub_epi64(inv, sign);
#else
                XSIMD_SPLIT_AVX(rhs);
                __m128i sign_low = _mm_cmpgt_epi64(_mm_setzero_si128(), rhs_low);
                __m128i sign_high = _mm_cmpgt_epi64(_mm_setzero_si128(), rhs_high);
                __m128i inv_low = _mm_xor_si128(rhs_low, sign_low);
                __m128i inv_high = _mm_xor_si128(rhs_high, sign_high);
                __m128i res_low = _mm_sub_epi64(inv_low, sign_low);
                __m128i res_high = _mm_sub_epi64(inv_high, sign_high);
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
                __m256i tmp1 = _mm256_shuffle_epi32(rhs, 0x0E);
                __m256i tmp2 = _mm256_add_epi64(rhs, tmp1);
                __m128i tmp3 = _mm256_extracti128_si256(tmp2, 1);
                __m128i res = _mm_add_epi64(_mm256_castsi256_si128(tmp2), tmp3);
#else
                XSIMD_SPLIT_AVX(rhs);
                __m128i tmp1 = _mm_shuffle_epi32(rhs_low, 0x0E);
                __m128i tmp2 = _mm_add_epi64(tmp1, rhs_low);
                __m128i tmp3 = _mm_shuffle_epi32(rhs_high, 0x0E);
                __m128i tmp4 = _mm_add_epi64(tmp3, rhs_high);
                __m128i res = _mm_add_epi64(tmp2, tmp4);
#endif
#if defined(__x86_64__)
                return _mm_cvtsi128_si64(res);
#else
                union {
                    int64_t i;
                    __m128i m;
                } u;
                _mm_storel_epi64(&u.m, res);
                return u.i;
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

    inline batch<int64_t, 4> operator<<(const batch<int64_t, 4>& lhs, int32_t rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
        return _mm256_slli_epi64(lhs, rhs);
#else
        XSIMD_SPLIT_AVX(lhs);
        __m128i res_low = _mm_slli_epi64(lhs_low, rhs);
        __m128i res_high = _mm_slli_epi64(lhs_high, rhs);
        XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
    }

    inline batch<int64_t, 4> operator>>(const batch<int64_t, 4>& lhs, int32_t rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
        return _mm256_srli_epi64(lhs, rhs);
#else
        XSIMD_SPLIT_AVX(lhs);
        __m128i res_low = _mm_srli_epi64(lhs_low, rhs);
        __m128i res_high = _mm_srli_epi64(lhs_high, rhs);
        XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
    }
}

#undef XSIMD_APPLY_SSE_FUNCTION
#undef XSIMD_RETURN_MERGED_SSE
#undef XSIMD_SPLIT_AVX

#endif
