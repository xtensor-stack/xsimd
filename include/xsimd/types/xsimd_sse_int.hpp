/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_SSE_INT_HPP
#define XSIMD_SSE_INT_HPP

#include "xsimd_base.hpp"

namespace xsimd
{

    /**********************
     * batch_bool<int, 4> *
     **********************/

    template <>
    class batch_bool<int, 4> : public simd_batch_bool<batch_bool<int, 4>>
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

    batch_bool<int, 4> operator&(const batch_bool<int, 4>& lhs, const batch_bool<int, 4>& rhs);
    batch_bool<int, 4> operator|(const batch_bool<int, 4>& lhs, const batch_bool<int, 4>& rhs);
    batch_bool<int, 4> operator^(const batch_bool<int, 4>& lhs, const batch_bool<int, 4>& rhs);
    batch_bool<int, 4> operator~(const batch_bool<int, 4>& rhs);

    batch_bool<int, 4> operator==(const batch_bool<int, 4>& lhs, const batch_bool<int, 4>& rhs);
    batch_bool<int, 4> operator!=(const batch_bool<int, 4>& lhs, const batch_bool<int, 4>& rhs);

    /*****************
     * batch<int, 4> *
     *****************/

    template <>
    struct simd_batch_traits<batch<int, 4>>
    {
        using value_type = int;
        static constexpr std::size_t size = 4;
    };

    template <>
    class batch<int, 4> : public simd_batch<batch<int, 4>>
    {

    public:

        batch();
        explicit batch(int i);
        batch(int i0, int i1, int i2, int i3);
        batch(const __m128i& rhs);
        batch& operator=(const __m128i& rhs);

        operator __m128i() const;

        batch& load_aligned(const int* src);
        batch& load_unaligned(const int* src);

        void store_aligned(int* dst) const;
        void store_unaligned(int* dst) const;

        int operator[](std::size_t index) const;

    private:

        __m128i m_value;
    };

    batch<int, 4> operator-(const batch<int, 4>& rhs);
    batch<int, 4> operator+(const batch<int, 4>& lhs, const batch<int, 4>& rhs);
    batch<int, 4> operator-(const batch<int, 4>& lhs, const batch<int, 4>& rhs);
    batch<int, 4> operator*(const batch<int, 4>& lhs, const batch<int, 4>& rhs);
    batch<int, 4> operator/(const batch<int, 4>& lhs, const batch<int, 4>& rhs);
    
    batch_bool<int, 4> operator==(const batch<int, 4>& lhs, const batch<int, 4>& rhs);
    batch_bool<int, 4> operator!=(const batch<int, 4>& lhs, const batch<int, 4>& rhs);
    batch_bool<int, 4> operator<(const batch<int, 4>& lhs, const batch<int, 4>& rhs);
    batch_bool<int, 4> operator<=(const batch<int, 4>& lhs, const batch<int, 4>& rhs);

    batch<int, 4> operator&(const batch<int, 4>& lhs, const batch<int, 4>& rhs);
    batch<int, 4> operator|(const batch<int, 4>& lhs, const batch<int, 4>& rhs);
    batch<int, 4> operator^(const batch<int, 4>& lhs, const batch<int, 4>& rhs);
    batch<int, 4> operator~(const batch<int, 4>& rhs);

    batch<int, 4> min(const batch<int, 4>& lhs, const batch<int, 4>& rhs);
    batch<int, 4> max(const batch<int, 4>& lhs, const batch<int, 4>& rhs);

    batch<int, 4> abs(const batch<int, 4>& rhs);

    batch<int, 4> fma(const batch<int, 4>& x, const batch<int, 4>& y, const batch<int, 4>& z);
    batch<int, 4> fms(const batch<int, 4>& x, const batch<int, 4>& y, const batch<int, 4>& z);
    batch<int, 4> fnma(const batch<int, 4>& x, const batch<int, 4>& y, const batch<int, 4>& z);
    batch<int, 4> fnms(const batch<int, 4>& x, const batch<int, 4>& y, const batch<int, 4>& z);

    int hadd(const batch<int, 4>& rhs);
    //batch<int, 4> haddp(const batch<int, 4>* row);

    batch<int, 4> select(const batch_bool<int, 4>& cond, const batch<int, 4>& a, const batch<int, 4>& b);

    batch<int, 4> operator<<(const batch<int, 4>& lhs, int rhs);
    batch<int, 4> operator>>(const batch<int, 4>& lhs, int rhs);

    /*************************************
     * batch_bool<int, 4> implementation *
     *************************************/

    inline batch_bool<int, 4>::batch_bool()
    {
    }

    inline batch_bool<int, 4>::batch_bool(bool b)
        : m_value(_mm_set1_epi32(-(int)b))
    {
    }

    inline batch_bool<int, 4>::batch_bool(bool b0, bool b1, bool b2, bool b3)
        : m_value(_mm_setr_epi32(-(int)b0, -(int)b1, -(int)b2, -(int)b3))
    {
    }

    inline batch_bool<int, 4>::batch_bool(const __m128i& rhs)
        : m_value(rhs)
    {
    }

    inline batch_bool<int, 4>& batch_bool<int, 4>::operator=(const __m128i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch_bool<int, 4>::operator __m128i() const
    {
        return m_value;
    }

    inline batch_bool<int, 4> operator&(const batch_bool<int, 4>& lhs, const batch_bool<int, 4>& rhs)
    {
        return _mm_and_si128(lhs, rhs);
    }

    inline batch_bool<int, 4> operator|(const batch_bool<int, 4>& lhs, const batch_bool<int, 4>& rhs)
    {
        return _mm_or_si128(lhs, rhs);
    }

    inline batch_bool<int, 4> operator^(const batch_bool<int, 4>& lhs, const batch_bool<int, 4>& rhs)
    {
        return _mm_xor_si128(lhs, rhs);
    }

    inline batch_bool<int, 4> operator~(const batch_bool<int, 4>& rhs)
    {
        return _mm_xor_si128(rhs, _mm_set1_epi32(-1));
    }

    inline batch_bool<int, 4> operator==(const batch_bool<int, 4>& lhs, const batch_bool<int, 4>& rhs)
    {
        return _mm_cmpeq_epi32(lhs, rhs);
    }

    inline batch_bool<int, 4> operator!=(const batch_bool<int, 4>& lhs, const batch_bool<int, 4>& rhs)
    {
        return ~(lhs == rhs);
    }

    /********************************
     * batch<int, 4> implementation *
     ********************************/

    inline batch<int, 4>::batch()
    {
    }

    inline batch<int, 4>::batch(int i)
        : m_value(_mm_set1_epi32(i))
    {
    }

    inline batch<int, 4>::batch(int i0, int i1, int i2, int i3)
        : m_value(_mm_setr_epi32(i0, i1, i2, i3))
    {
    }

    inline batch<int, 4>::batch(const __m128i& rhs)
        : m_value(rhs)
    {
    }

    inline batch<int, 4>& batch<int, 4>::operator=(const __m128i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<int, 4>::operator __m128i() const
    {
        return m_value;
    }

    inline batch<int, 4>& batch<int, 4>::load_aligned(const int* src)
    {
        m_value = _mm_load_si128((__m128i const*)src);
        return *this;
    }

    inline batch<int, 4>& batch<int, 4>::load_unaligned(const int* src)
    {
        m_value = _mm_loadu_si128((__m128i const*)src);
        return *this;
    }

    inline void batch<int, 4>::store_aligned(int* dst) const
    {
        _mm_store_si128((__m128i*)dst, m_value);
    }

    inline void batch<int, 4>::store_unaligned(int* dst) const
    {
        _mm_storeu_si128((__m128i*)dst, m_value);
    }

    inline int batch<int, 4>::operator[](std::size_t index) const
    {
        alignas(16) int x[4];
        store_aligned(x);
        return x[index & 3];
    }

    inline batch<int, 4> operator-(const batch<int, 4>& rhs)
    {
        return _mm_sub_epi32(_mm_setzero_si128(), rhs);
    }

    inline batch<int, 4> operator+(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
    {
        return _mm_add_epi32(lhs, rhs);
    }

    inline batch<int, 4> operator-(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
    {
        return _mm_sub_epi32(lhs, rhs);
    }

    inline batch<int, 4> operator*(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
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

    /*inline batch<int, 4> operator/(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
    {
    }*/

    inline batch_bool<int, 4> operator==(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
    {
        return _mm_cmpeq_epi32(lhs, rhs);
    }

    inline batch_bool<int, 4> operator!=(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
    {
        return ~(lhs == rhs);
    }

    inline batch_bool<int, 4> operator<(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
    {
        return _mm_cmplt_epi32(lhs, rhs);
    }

    inline batch_bool<int, 4> operator<=(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
    {
        return ~(rhs < lhs);
    }

    inline batch<int, 4> operator&(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
    {
        return _mm_and_si128(lhs, rhs);
    }

    inline batch<int, 4> operator|(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
    {
        return _mm_or_si128(lhs, rhs);
    }

    inline batch<int, 4> operator^(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
    {
        return _mm_xor_si128(lhs, rhs);
    }

    inline batch<int, 4> operator~(const batch<int, 4>& rhs)
    {
        return _mm_xor_si128(rhs, _mm_set1_epi32(-1));
    }

    inline batch<int, 4> min(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        return _mm_min_epi32(lhs, rhs);
#else
        __m128i greater = _mm_cmpgt_epi32(lhs, rhs);
        return selectb(greater, rhs, lhs);
#endif
    }

    inline batch<int, 4> max(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        return _mm_max_epi32(lhs, rhs);
#else
        __m128i greater = _mm_cmpgt_epi32(lhs, rhs);
        return selectb(greater, lhs, rhs);
#endif
    }

    inline batch<int, 4> abs(const batch<int, 4>& rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSSE3_VERSION
        return _mm_sign_epi32(rhs, rhs);
#else
        __m128i sign = _mm_srai_epi32(rhs, 31);
        __m128i inv = _mm_xor_si128(rhs, sign);
        return _mm_sub_epi32(inv, sign);
#endif
    }

    inline batch<int, 4> fma(const batch<int, 4>& x, const batch<int, 4>& y, const batch<int, 4>& z)
    {
        return x * y + z;
    }

    inline batch<int, 4> fms(const batch<int, 4>& x, const batch<int, 4>& y, const batch<int, 4>& z)
    {
        return x * y - z;
    }

    inline batch<int, 4> fnma(const batch<int, 4>& x, const batch<int, 4>& y, const batch<int, 4>& z)
    {
        return -x * y + z;
    }

    inline batch<int, 4> fnms(const batch<int, 4>& x, const batch<int, 4>& y, const batch<int, 4>& z)
    {
        return -x * y - z;
    }

    inline int hadd(const batch<int, 4>& rhs)
    {
#if  XSIMD_X86_INSTR_SET >= XSIMD_X86_SSSE3_VERSION
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

    //inline batch<int, 4> haddp(const batch<int, 4>* row);

    inline batch<int, 4> select(const batch_bool<int, 4>& cond, const batch<int, 4>& a, const batch<int, 4>& b)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        return _mm_blendv_epi8(b, a, cond);
#else
        return _mm_or_si128(_mm_and_si128(cond, a), _mm_andnot_si128(s, b));
#endif
    }

    inline batch<int, 4> operator<<(const batch<int, 4>& lhs, int rhs)
    {
        return _mm_slli_epi32(lhs, rhs);
    }

    inline batch<int, 4> operator>>(const batch<int, 4>& lhs, int rhs)
    {
        return _mm_srli_epi32(lhs, rhs);
    }

}

#endif
