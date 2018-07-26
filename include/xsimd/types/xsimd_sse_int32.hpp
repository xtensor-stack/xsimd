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
        using value_type = int32_t;
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

        bool operator[](std::size_t index) const;

    private:

        __m128i m_value;
    };

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

        using self_type = batch<int32_t, 4>;
        using base_type = simd_batch<self_type>;

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

        batch& load_aligned(const int8_t* src);
        batch& load_unaligned(const int8_t* src);

        batch& load_aligned(const uint8_t* src);
        batch& load_unaligned(const uint8_t* src);

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

        __m128i m_value;
    };

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

    inline bool batch_bool<int32_t, 4>::operator[](std::size_t index) const
    {
        alignas(16) int32_t x[4];
        _mm_store_si128((__m128i*)x, m_value);
        return static_cast<bool>(x[index & 3]);
    }

    namespace detail
    {
        template <>
        struct batch_bool_kernel<int32_t, 4>
        {
            using batch_type = batch_bool<int32_t, 4>;

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_and_si128(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_or_si128(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_xor_si128(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return _mm_xor_si128(rhs, _mm_set1_epi32(-1));
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_andnot_si128(lhs, rhs);
            }

            static batch_type equal(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_cmpeq_epi32(lhs, rhs);
            }

            static batch_type not_equal(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(lhs == rhs);
            }

            static bool all(const batch_type& rhs)
            {
                return _mm_movemask_epi8(rhs) == 0xFFFF;
            }

            static bool any(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
                return !_mm_testz_si128(rhs, rhs);
#else
                return _mm_movemask_epi8(rhs) != 0;
#endif
            }
        };
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

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_aligned(const int8_t* src)
    {
        __m128i tmp = _mm_loadl_epi64((const __m128i*)src);
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        m_value = _mm_cvtepi8_epi32(tmp);
#else
        __m128i mask = _mm_cmplt_epi8(tmp, _mm_set1_epi8(0));
        __m128i tmp1 = _mm_unpacklo_epi8(tmp, mask);
        mask = _mm_cmplt_epi16(tmp1, _mm_set1_epi16(0));
        m_value = _mm_unpacklo_epi16(tmp1, mask);
#endif
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_unaligned(const int8_t* src)
    {
        return load_aligned(src);
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_aligned(const uint8_t* src)
    {
        __m128i tmp = _mm_loadl_epi64((const __m128i*)src);
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        m_value = _mm_cvtepu8_epi32(tmp);
#else
        __m128i tmp2 = _mm_unpacklo_epi8(tmp, _mm_set1_epi8(0));
        m_value = _mm_unpacklo_epi16(tmp2, _mm_set1_epi16(0));
#endif
        return *this;
    }

    inline batch<int32_t, 4>& batch<int32_t, 4>::load_unaligned(const uint8_t* src)
    {
        return load_aligned(src);
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

    inline void batch<int32_t, 4>::store_aligned(int8_t* dst) const
    {
        __m128i tmp1 = _mm_packs_epi32(m_value, m_value);
        __m128i tmp2 = _mm_packs_epi16(tmp1, tmp1);
        _mm_storel_epi64((__m128i*)dst, tmp2);
    }

    inline void batch<int32_t, 4>::store_unaligned(int8_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int32_t, 4>::store_aligned(uint8_t* dst) const
    {
        __m128i tmp1 = _mm_packs_epi32(m_value, m_value);
        __m128i tmp2 = _mm_packus_epi16(tmp1, tmp1);
        _mm_storel_epi64((__m128i*)dst, tmp2);
    }

    inline void batch<int32_t, 4>::store_unaligned(uint8_t* dst) const
    {
        store_aligned(dst);
    }

    inline int32_t batch<int32_t, 4>::operator[](std::size_t index) const
    {
        alignas(16) int32_t x[4];
        store_aligned(x);
        return x[index & 3];
    }

    namespace detail
    {
        template <>
        struct batch_kernel<int32_t, 4>
        {
            using batch_type = batch<int32_t, 4>;
            using value_type = int32_t;
            using batch_bool_type = batch_bool<int32_t, 4>;

            static batch_type neg(const batch_type& rhs)
            {
                return _mm_sub_epi32(_mm_setzero_si128(), rhs);
            }

            static batch_type add(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_add_epi32(lhs, rhs);
            }

            static batch_type sub(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_sub_epi32(lhs, rhs);
            }

            static batch_type mul(const batch_type& lhs, const batch_type& rhs)
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

            static batch_type div(const batch_type& lhs, const batch_type& rhs)
            {
#if defined(XSIMD_FAST_INTEGER_DIVISION)
                return _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(lhs), _mm_cvtepi32_ps(rhs)));
#else
                alignas(16) int32_t tmp_lhs[4], tmp_rhs[4], tmp_res[4];
                lhs.store_aligned(tmp_lhs);
                rhs.store_aligned(tmp_rhs);
                tmp_res[0] = tmp_lhs[0] / tmp_rhs[0];
                tmp_res[1] = tmp_lhs[1] / tmp_rhs[1];
                tmp_res[2] = tmp_lhs[2] / tmp_rhs[2];
                tmp_res[3] = tmp_lhs[3] / tmp_rhs[3];
                return batch_type(&tmp_res[0], aligned_mode());
#endif
            }

            static batch_bool_type eq(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_cmpeq_epi32(lhs, rhs);
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(lhs == rhs);
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_cmplt_epi32(lhs, rhs);
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(rhs < lhs);
            }

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_and_si128(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_or_si128(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_xor_si128(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return _mm_xor_si128(rhs, _mm_set1_epi32(-1));
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_andnot_si128(lhs, rhs);
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
                return _mm_min_epi32(lhs, rhs);
#else
                __m128i greater = _mm_cmpgt_epi32(lhs, rhs);
                return select(greater, rhs, lhs);
#endif
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
                return _mm_max_epi32(lhs, rhs);
#else
                __m128i greater = _mm_cmpgt_epi32(lhs, rhs);
                return select(greater, lhs, rhs);
#endif
            }

            static batch_type abs(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSSE3_VERSION
                return _mm_sign_epi32(rhs, rhs);
#else
                __m128i sign = _mm_srai_epi32(rhs, 31);
                __m128i inv = _mm_xor_si128(rhs, sign);
                return _mm_sub_epi32(inv, sign);
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

            static batch_type select(const batch_bool_type& cond, const batch_type& a, const batch_type& b)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
                return _mm_blendv_epi8(b, a, cond);
#else
                return _mm_or_si128(_mm_and_si128(cond, a), _mm_andnot_si128(cond, b));
#endif
            }
        };
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
