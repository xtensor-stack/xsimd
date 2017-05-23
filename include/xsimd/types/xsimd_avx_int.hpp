/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX_INT_HPP
#define XSIMD_AVX_INT_HPP

#include "xsimd_base.hpp"

namespace xsimd
{

    /**********************
     * batch_bool<int, 8> *
     **********************/

    template <>
    class batch_bool<int, 8> : public simd_batch_bool<batch_bool<int, 8>>
    {

    public:

        batch_bool();
        explicit batch_bool(bool b);
        batch_bool(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7);
        batch_bool(const __m256i& rhs);
        batch_bool& operator=(const __m256i& rhs);

        operator __m256i() const;

    private:

        __m256i m_value;
    };

    batch_bool<int, 8> operator&(const batch_bool<int, 8>& lhs, const batch_bool<int, 8>& rhs);
    batch_bool<int, 8> operator|(const batch_bool<int, 8>& lhs, const batch_bool<int, 8>& rhs);
    batch_bool<int, 8> operator^(const batch_bool<int, 8>& lhs, const batch_bool<int, 8>& rhs);
    batch_bool<int, 8> operator~(const batch_bool<int, 8>& rhs);

    batch_bool<int, 8> operator==(const batch_bool<int, 8>& lhs, const batch_bool<int, 8>& rhs);
    batch_bool<int, 8> operator!=(const batch_bool<int, 8>& lhs, const batch_bool<int, 8>& rhs);

    /*****************
     * batch<int, 8> *
     *****************/

    template <>
    struct simd_batch_traits<batch<int, 8>>
    {
        using value_type = int;
        static constexpr std::size_t size = 8;
    };

    template <>
    class batch<int, 8> : public simd_batch<batch<int, 8>>
    {

    public:

        batch();
        explicit batch(int i);
        batch(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7);
        batch(const __m256i& rhs);
        batch& operator=(const __m256i& rhs);

        operator __m256i() const;

        batch& load_aligned(const int* src);
        batch& load_unaligned(const int* src);

        void store_aligned(int* dst) const;
        void store_unaligned(int* dst) const;

    private:

        __m256i m_value;
    };

    batch<int, 8> operator-(const batch<int, 8>& rhs);
    batch<int, 8> operator+(const batch<int, 8>& lhs, const batch<int, 8>& rhs);
    batch<int, 8> operator-(const batch<int, 8>& lhs, const batch<int, 8>& rhs);
    batch<int, 8> operator*(const batch<int, 8>& lhs, const batch<int, 8>& rhs);
    batch<int, 8> operator/(const batch<int, 8>& lhs, const batch<int, 8>& rhs);

    batch_bool<int, 8> operator==(const batch<int, 8>& lhs, const batch<int, 8>& rhs);
    batch_bool<int, 8> operator!=(const batch<int, 8>& lhs, const batch<int, 8>& rhs);
    batch_bool<int, 8> operator<(const batch<int, 8>& lhs, const batch<int, 8>& rhs);
    batch_bool<int, 8> operator<=(const batch<int, 8>& lhs, const batch<int, 8>& rhs);

    batch<int, 8> operator&(const batch<int, 8>& lhs, const batch<int, 8>& rhs);
    batch<int, 8> operator|(const batch<int, 8>& lhs, const batch<int, 8>& rhs);
    batch<int, 8> operator^(const batch<int, 8>& lhs, const batch<int, 8>& rhs);
    batch<int, 8> operator~(const batch<int, 8>& rhs);

    batch<int, 8> min(const batch<int, 8>& lhs, const batch<int, 8>& rhs);
    batch<int, 8> max(const batch<int, 8>& lhs, const batch<int, 8>& rhs);

    batch<int, 8> abs(const batch<int, 8>& rhs);

    batch<int, 8> fma(const batch<int, 8>& x, const batch<int, 8>& y, const batch<int, 8>& z);
    batch<int, 8> fms(const batch<int, 8>& x, const batch<int, 8>& y, const batch<int, 8>& z);
    batch<int, 8> fnma(const batch<int, 8>& x, const batch<int, 8>& y, const batch<int, 8>& z);
    batch<int, 8> fnms(const batch<int, 8>& x, const batch<int, 8>& y, const batch<int, 8>& z);

    int hadd(const batch<int, 8>& rhs);
    //batch<int, 4> haddp(const batch<int, 4>* row);

    batch<int, 8> select(const batch_bool<int, 8>& cond, const batch<int, 8>& a, const batch<int, 8>& b);

    /*************************************
     * batch_bool<int, 8> implementation *
     *************************************/

    inline batch_bool<int, 8>::batch_bool()
    {
    }

    inline batch_bool<int, 8>::batch_bool(bool b)
        : m_value(_mm256_set1_epi32(-(int)b))
    {
    }

    inline batch_bool<int, 8>::batch_bool(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7)
        : m_value(_mm256_setr_epi32(-(int)b0, -(int)b1, -(int)b2, -(int)b3, -(int)b4, -(int)b5, -(int)b6, -(int)b7))
    {
    }

    inline batch_bool<int, 8>::batch_bool(const __m256i& rhs)
        : m_value(rhs)
    {
    }

    inline batch_bool<int, 8>& batch_bool<int, 8>::operator=(const __m256i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch_bool<int, 8>::operator __m256i() const
    {
        return m_value;
    }

    inline batch_bool<int, 8> operator&(const batch_bool<int, 8>& lhs, const batch_bool<int, 8>& rhs)
    {
        return _mm256_and_si256(lhs, rhs);
    }

    inline batch_bool<int, 8> operator|(const batch_bool<int, 8>& lhs, const batch_bool<int, 8>& rhs)
    {
        return _mm256_or_si256(lhs, rhs);
    }

    inline batch_bool<int, 8> operator^(const batch_bool<int, 8>& lhs, const batch_bool<int, 8>& rhs)
    {
        return _mm256_xor_si256(lhs, rhs);
    }

    inline batch_bool<int, 8> operator~(const batch_bool<int, 8>& rhs)
    {
        return _mm256_xor_si256(rhs, _mm256_set1_epi32(-1));
    }

    inline batch_bool<int, 8> operator==(const batch_bool<int, 8>& lhs, const batch_bool<int, 8>& rhs)
    {
        return _mm256_cmpeq_epi32(lhs, rhs);
    }

    inline batch_bool<int, 8> operator!=(const batch_bool<int, 8>& lhs, const batch_bool<int, 8>& rhs)
    {
        return ~(lhs == rhs);
    }

    /********************************
     * batch<int, 8> implementation *
     ********************************/

    inline batch<int, 8>::batch()
    {
    }

    inline batch<int, 8>::batch(int i)
        : m_value(_mm256_set1_epi32(i))
    {
    }

    inline batch<int, 8>::batch(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7)
        : m_value(_mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7))
    {
    }

    inline batch<int, 8>::batch(const __m256i& rhs)
        : m_value(rhs)
    {
    }

    inline batch<int, 8>& batch<int, 8>::operator=(const __m256i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<int, 8>::operator __m256i() const
    {
        return m_value;
    }

    inline batch<int, 8>& batch<int, 8>::load_aligned(const int* src)
    {
        m_value = _mm256_load_si256((__m256i const*)src);
        return *this;
    }

    inline batch<int, 8>& batch<int, 8>::load_unaligned(const int* src)
    {
        m_value = _mm256_loadu_si256((__m256i const*)src);
        return *this;
    }

    inline void batch<int, 8>::store_aligned(int* dst) const
    {
        _mm256_store_si256((__m256i*)dst, m_value);
    }

    inline void batch<int, 8>::store_unaligned(int* dst) const
    {
        _mm256_storeu_si256((__m256i*)dst, m_value);
    }

    inline batch<int, 8> operator-(const batch<int, 8>& rhs)
    {
        return _mm256_sub_epi32(_mm256_setzero_si256(), rhs);
    }

    inline batch<int, 8> operator+(const batch<int, 8>& lhs, const batch<int, 8>& rhs)
    {
        return _mm256_add_epi32(lhs, rhs);
    }

    inline batch<int, 8> operator-(const batch<int, 8>& lhs, const batch<int, 8>& rhs)
    {
        return _mm256_sub_epi32(lhs, rhs);
    }

    inline batch<int, 8> operator*(const batch<int, 8>& lhs, const batch<int, 8>& rhs)
    {
        return _mm256_mullo_epi32(lhs, rhs);
    }

    /*inline batch<int, 4> operator/(const batch<int, 4>& lhs, const batch<int, 4>& rhs)
    {
    }*/

    inline batch_bool<int, 8> operator==(const batch<int, 8>& lhs, const batch<int, 8>& rhs)
    {
        return _mm256_cmpeq_epi32(lhs, rhs);
    }

    inline batch_bool<int, 8> operator!=(const batch<int, 8>& lhs, const batch<int, 8>& rhs)
    {
        return ~(lhs == rhs);
    }

    inline batch_bool<int, 8> operator<(const batch<int, 8>& lhs, const batch<int, 8>& rhs)
    {
        return _mm256_cmpgt_epi32(rhs, lhs);
    }

    inline batch_bool<int, 8> operator<=(const batch<int, 8>& lhs, const batch<int, 8>& rhs)
    {
        return ~(rhs < lhs);
    }

    inline batch<int, 8> operator&(const batch<int, 8>& lhs, const batch<int, 8>& rhs)
    {
        return _mm256_and_si256(lhs, rhs);
    }

    inline batch<int, 8> operator|(const batch<int, 8>& lhs, const batch<int, 8>& rhs)
    {
        return _mm256_or_si256(lhs, rhs);
    }

    inline batch<int, 8> operator^(const batch<int, 8>& lhs, const batch<int, 8>& rhs)
    {
        return _mm256_xor_si256(lhs, rhs);
    }

    inline batch<int, 8> operator~(const batch<int, 8>& rhs)
    {
        return _mm256_xor_si256(rhs, _mm256_set1_epi32(-1));
    }

    inline batch<int, 8> min(const batch<int, 8>& lhs, const batch<int, 8>& rhs)
    {
        return _mm256_min_epi32(lhs, rhs);
    }

    inline batch<int, 8> max(const batch<int, 8>& lhs, const batch<int, 8>& rhs)
    {
        return _mm256_max_epi32(lhs, rhs);
    }

    inline batch<int, 8> abs(const batch<int, 8>& rhs)
    {
        return _mm256_sign_epi32(rhs, rhs);
    }

    inline batch<int, 8> fma(const batch<int, 8>& x, const batch<int, 8>& y, const batch<int, 8>& z)
    {
        return x * y + z;
    }

    inline batch<int, 8> fms(const batch<int, 8>& x, const batch<int, 8>& y, const batch<int, 8>& z)
    {
        return x * y - z;
    }
    
    inline batch<int, 8> fnma(const batch<int, 8>& x, const batch<int, 8>& y, const batch<int, 8>& z)
    {
        return -x * y + z;
    }

    inline batch<int, 8> fnms(const batch<int, 8>& x, const batch<int, 8>& y, const batch<int, 8>& z)
    {
        return -x * y - z;
    }

    inline int hadd(const batch<int, 8>& rhs)
    {
        __m256i tmp1 = _mm256_hadd_epi32(rhs, rhs);
        __m256i tmp2 = _mm256_hadd_epi32(tmp1, tmp1);
        __m128i tmp3 = _mm256_extracti128_si256(tmp2, 1);
        __m128i tmp4 = _mm_add_epi32(_mm256_castsi256_si128(tmp2), tmp3);
        return _mm_cvtsi128_si32(tmp4);
    }

    //inline batch<int, 4> haddp(const batch<int, 4>* row);

    inline batch<int, 8> select(const batch_bool<int, 8>& cond, const batch<int, 8>& a, const batch<int, 8>& b)
    {
        return _mm256_blendv_epi8(b, a, cond);
    }

}

#endif

