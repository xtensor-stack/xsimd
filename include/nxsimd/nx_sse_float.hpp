//
// Copyright (c) 2012 - 2016 Johan Mabille
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
//

#ifndef NX_SSE_FLOAT_HPP
#define NX_SSE_FLOAT_HPP

#include "nx_simd_base.hpp"

namespace nxsimd
{

    class vector4fb : public simd_vector_bool<vector4fb>
    {

    public:

        vector4fb();
        vector4fb(bool b);
        vector4fb(bool b0, bool b1, bool b2, bool b3);
        vector4fb(const __m128& rhs);
        vector4fb& operator=(const __m128& rhs);

        operator __128() const;

    private:

        __m128 m_value;
    };

    vector4fb operator&(const vector4fb& lhs, const vector4fb& rhs);
    vector4fb operator|(const vector4fb& lhs, const vector4fb& rhs);
    vector4fb operator^(const vector4fb& lhs, const vector4fb& rhs);
    vector4fb operator~(const vector4fb& rhs);

    vector4fb operator==(const vector4fb& lhs, const vector4fb& rhs);
    vector4fb operator!=(const vector4fb& lhs, const vector4fb& rhs);

    class vector4f;

    template <>
    struct simd_vector_traits<vector4f>
    {
        using value_type = float;
        using vector_bool = vector4fb;
    };

    class vector4f : public simd_vector<vector4f>
    {

    public:

        vector4f();
        vector4f(float f);
        vector4f(float f0, float f1, float f2, float f3);
        vector4f(const __m128& rhs);
        vector4f& operator=(const __m128& rhs);

        vector4f operator __m128() const;

        vector4f& load_aligned(const float* src);
        vector4f& load_unaligned(const float* src);

        void store_aligned(float* dst);
        void store_unaligned(float* dst);

    private:

        __m128 m_value;
    };

    vector4f operator-(const vector4f& rhs);
    vector4f operator+(const vector4f& lhs, const vector4f& rhs);
    vector4f operator-(const vector4f& lhs, const vector4f& rhs);
    vector4f operator*(const vector4f& lhs, const vector4f& rhs);
    vector4f operator/(const vector4f& lhs, const vector4f& rhs);
    
    vector4fb operator==(const vector4f& lhs, const vector4f& rhs);
    vector4fb operator!=(const vector4f& lhs, const vector4f& rhs);
    vector4fb operator<(const vector4f& lhs, const vector4f& rhs);
    vector4fb operator<=(const vector4f& lhs, const vector4f& rhs);

    vector4f operator&(const vector4f& lhs, const vector4f& rhs);
    vector4f operator|(const vector4f& lhs, const vector4f& rhs);
    vector4f operator^(const vector4f& lhs, const vector4f& rhs);
    vector4f operator~(const vector4f& rhs);

    float hadd(const vector4f& rhs);
    vector4f haddp(const vector4f* row);

    vector4f select(const vector4fb& cond, const vector4f& a, const vector4f& b);


    /******************************
     * vector4fb implementation
     ******************************/

    inline vector4fb::vector4fb()
    {
    }

    inline vector4fb::vector4fb(bool b)
        : m_value(_mm_castsi128_ps(_mm_set1_epi32(-(int)b)))
    {
    }

    inline vector4fb::vector4fb(bool b0, bool b1, bool b2, bool b3)
        : m_value(_mm_castsi128_ps(_mm_setr_epi32(-(int)b0, -(int)b1, -(int)b2, -(int)b3)))
    {
    }

    inline vector4fb::vector4fb(const __m128& rhs)
        : m_value(rhs)
    {
    }

    inline vector4fb& vector4fb::operator=(const __m128& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline vector4fb::operator __128() const
    {
        return m_value;
    }

    inline vector4fb operator&(const vector4fb& lhs, const vector4fb& rhs)
    {
        return _mm_and_ps(lhs, rhs);
    }

    inline vector4fb operator|(const vector4fb& lhs, const vector4fb& rhs)
    {
        return _mm_or_ps(lhs, rhs);
    }

    inline vector4fb operator^(const vector4fb& lhs, const vector4fb& rhs)
    {
        return _mm_xor_ps(lhs, rhs);
    }

    inline vector4fb operator~(const vector4fb& rhs)
    {
        return _mm_xor_ps(rhs, _mm_castsi128_ps(_mm_set1_epi32(-1)));
    }

    inline vector4fb operator==(const vector4fb& lhs, const vector4fb& rhs)
    {
        return _mm_cmpeq_ps(lhs, rhs);
    }

    inline vector4fb operator!=(const vector4fb& lhs, const vector4fb& rhs)
    {
        return _mm_cmpneq(lhs, rhs);
    }


    /*****************************
     * vector4f implementation
     *****************************/

    inline vector4f::vector4f()
    {
    }

    inline vector4f::vector4f(float f)
        : m_value(_mm_set1_ps(f))
    {
    }
    
    inline vector4f::vector4f(float f0, float f1, float f2, float f3)
        : m_value(_mm_setr_ps(f0, f1, f2, f3))
    {
    }

    inline vector4f::vector4f(const __m128& rhs)
        : m_value(rhs)
    {
    }

    inline vector4f& vetcor4f::operator=(const __m128& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline vector4f vector4f::operator __m128() const
    {
        return m_value;
    }

    inline vector4f& vector4f::load_aligned(const float* src)
    {
        m_value = _mm_load_ps(src);
        return *this;
    }

    inline vector4f& vector4f::load_unaligned(const float* src)
    {
        m_value = _mm_loadu_ps(src);
        return *this;
    }

    inline void vector4f::store_aligned(float* dst)
    {
        _mm_store_ps(dst, m_value);
    }

    inline void vector4f::store_unaligned(float* dst)
    {
        _mm_storeu_ps(dst, m_value);
    }

    inline vector4f operator-(const vector4f& rhs)
    {
        return _mm_xor_ps(rhs, _mm_castsi128_ps(_mm_set1_epi32(0x80000000)));
    }

    inline vector4f operator+(const vector4f& lhs, const vector4f& rhs)
    {
        return _mm_add_ps(lhs, rhs);
    }

    inline vector4f operator-(const vector4f& lhs, const vector4f& rhs)
    {
        return _mm_sub_ps(lhs, rhs);
    }
    
    inline vector4f operator*(const vector4f& lhs, const vector4f& rhs)
    {
        return _mm_mul_ps(lhs, rhs);
    }

    inline vector4f operator/(const vector4f& lhs, const vector4f& rhs)
    {
        return _mm_div_ps(lhs, rhs);
    }
    
    inline vector4fb operator==(const vector4f& lhs, const vector4f& rhs)
    {
        return _mm_cmpeq_ps(lhs, rhs);
    }

    inline vector4fb operator!=(const vector4f& lhs, const vector4f& rhs)
    {
        return _mm_cmpneq_ps(lhs, rhs);
    }

    inline vector4fb operator<(const vector4f& lhs, const vector4f& rhs)
    {
        return _mm_cmplt_ps(lhs, rhs);
    }

    inline vector4fb operator<=(const vector4f& lhs, const vector4f& rhs)
    {
        return _mm_cmple_ps(lhs, rhs);
    }

    inline vector4f operator&(const vector4f& lhs, const vector4f& rhs)
    {
        return _mm_and_ps(lhs, rhs);
    }

    inline vector4f operator|(const vector4f& lhs, const vector4f& rhs)
    {
        return _mm_or_ps(lhs, rhs);
    }

    inline vector4f operator^(const vector4f& lhs, const vector4f& rhs)
    {
        return _mm_xor_ps(lhs, rhs);
    }

    inline vector4f operator~(const vector4f& rhs)
    {
        return _mm_xor_ps(rhs, _mm_castsi128_ps(_mm_set1_epi32(-1)));
    }

    inline float hadd(const vector4f& rhs)
    {
#if SSE_INSTR_SET >= 3  // SSE3
        __m128 tmp0 = _mm_hadd_ps(rhs, rhs);
        __m128 tmp1 = _mm_hadd_ps(tmp0, tmp0);
#else
        __m128 tmp0 = _mm_add_ps(rhs, _mm_movehl_ps(rhs, rhs));
        __m128 tmp1 = _mm_add_ss(tmp0, _mm_shuffle_ps(tmp0, tmp0, 1));
#endif
        return _mm_cvtss_f32(tmp1);
    }
    
    inline vector4f haddp(const vector4f* row)
    {
#if SSE_INSTR_SET >= 3  // SSE3
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

    inline vector4f select(const vector4fb& cond, const vector4f& a, const vector4f& b)
    {
#if SSE_INSTR_SET >= 5  // SSE 4.1
        return _mm_blendv_ps(b, a, cond);
#else
        return _mm_or_ps(_mm_and_ps(cond, a), _mm_andnot_ps(cond, b));
#endif
    }
}

#endif

