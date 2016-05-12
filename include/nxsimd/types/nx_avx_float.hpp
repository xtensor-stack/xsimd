//
// Copyright (c) 2016 Johan Mabille
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
//

#ifndef NX_AVX_FLOAT_HPP
#define NX_AVX_FLOAT_HPP

#include "nx_simd_base.hpp"

namespace nxsimd
{

    class vector8fb : public simd_vector_bool<vector8fb>
    {

    public:

        vector8fb();
        explicit vector8fb(bool b);
        vector8fb(bool b0, bool b1, bool b2, bool b3,
                  bool b4, bool b5, bool b6, bool b7);
        vector8fb(const __m256& rhs);
        vector8fb& operator=(const __m256& rhs);

        operator __m256() const;

    private:

        __m256 m_value;
    };

    vector8fb operator&(const vector8fb& lhs, const vector8fb& rhs);
    vector8fb operator|(const vector8fb& lhs, const vector8fb& rhs);
    vector8fb operator^(const vector8fb& lhs, const vector8fb& rhs);
    vector8fb operator~(const vector8fb& rhs);

    vector8fb operator==(const vector8fb& lhs, const vector8fb& rhs);
    vector8fb operator!=(const vector8fb& lhs, const vector8fb& rhs);

    class vector8f;

    template <>
    struct simd_vector_traits<vector8f>
    {
        using value_type = float;
        using vector_bool = vector8fb;
    };

    class vector8f : public simd_vector<vector8f>
    {

    public:

        vector8f();
        explicit vector8f(float f);
        vector8f(float f0, float f1, float f2, float f3,
                 float f4, float f5, float f6, float f7);
        vector8f(const __m256& rhs);
        vector8f& operator=(const __m256& rhs);

        operator __m256() const;

        vector8f& load_aligned(const float* src);
        vector8f& load_unaligned(const float* src);

        void store_aligned(float* dst) const;
        void store_unaligned(float* dst) const;

    private:

        __m256 m_value;
    };

    vector8f operator-(const vector8f& rhs);
    vector8f operator+(const vector8f& lhs, const vector8f& rhs);
    vector8f operator-(const vector8f& lhs, const vector8f& rhs);
    vector8f operator*(const vector8f& lhs, const vector8f& rhs);
    vector8f operator/(const vector8f& lhs, const vector8f& rhs);
    
    vector8fb operator==(const vector8f& lhs, const vector8f& rhs);
    vector8fb operator!=(const vector8f& lhs, const vector8f& rhs);
    vector8fb operator<(const vector8f& lhs, const vector8f& rhs);
    vector8fb operator<=(const vector8f& lhs, const vector8f& rhs);

    vector8f operator&(const vector8f& lhs, const vector8f& rhs);
    vector8f operator|(const vector8f& lhs, const vector8f& rhs);
    vector8f operator^(const vector8f& lhs, const vector8f& rhs);
    vector8f operator~(const vector8f& rhs);

    vector8f min(const vector8f& lhs, const vector8f& rhs);
    vector8f max(const vector8f& lhs, const vector8f& rhs);

    vector8f abs(const vector8f& rhs);

    vector8f fma(const vector8f& x, const vector8f& y, const vector8f& z);

    vector8f sqrt(const vector8f& rhs);

    float hadd(const vector8f& rhs);
    vector8f haddp(const vector8f* row);

    vector8f select(const vector8fb& cond, const vector8f& a, const vector8f& b);


    /******************************
     * vector8fb implementation
     ******************************/

    inline vector8fb::vector8fb()
    {
    }

    inline vector8fb::vector8fb(bool b)
        : m_value(_mm256_castsi256_ps(_mm256_set1_epi32(-(int)b)))
    {
    }

    inline vector8fb::vector8fb(bool b0, bool b1, bool b2, bool b3,
                                bool b4, bool b5, bool b6, bool b7)
            : m_value(_mm256_castsi256_ps(
                  _mm256_setr_epi32(-(int)b0, -(int)b1, -(int)b2, -(int)b3,
                                    -(int)b4, -(int)b5, -(int)b6, -(int)b7)))
    {
    }

    inline vector8fb::vector8fb(const __m256& rhs)
        : m_value(rhs)
    {
    }

    inline vector8fb& vector8fb::operator=(const __m256& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline vector8fb::operator __m256() const
    {
        return m_value;
    }

    inline vector8fb operator&(const vector8fb& lhs, const vector8fb& rhs)
    {
        return _mm256_and_ps(lhs, rhs);
    }

    inline vector8fb operator|(const vector8fb& lhs, const vector8fb& rhs)
    {
        return _mm256_or_ps(lhs, rhs);
    }

    inline vector8fb operator^(const vector8fb& lhs, const vector8fb& rhs)
    {
        return _mm256_xor_ps(lhs, rhs);
    }

    inline vector8fb operator~(const vector8fb& rhs)
    {
        return _mm256_xor_ps(rhs, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
    }

    inline vector8fb operator==(const vector8fb& lhs, const vector8fb& rhs)
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_EQ_OQ);
    }

    inline vector8fb operator!=(const vector8fb& lhs, const vector8fb& rhs)
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NEQ_OQ);
    }


    /*****************************
     * vector8f implementation
     *****************************/

    inline vector8f::vector8f()
    {
    }

    inline vector8f::vector8f(float f)
        : m_value(_mm256_set1_ps(f))
    {
    }
    inline vector8f::vector8f(float f0, float f1, float f2, float f3,
                              float f4, float f5, float f6, float f7)
        : m_value(_mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7))
    {
    }

    inline vector8f::vector8f(const __m256& rhs)
        : m_value(rhs)
    {
    }

    inline vector8f& vector8f::operator=(const __m256& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline vector8f::operator __m256() const
    {
        return m_value;
    }

    inline vector8f& vector8f::load_aligned(const float* src)
    {
        m_value = _mm256_load_ps(src);
        return *this;
    }

    inline vector8f& vector8f::load_unaligned(const float* src)
    {
        m_value = _mm256_loadu_ps(src);
        return *this;
    }

    inline void vector8f::store_aligned(float* dst) const
    {
        _mm256_store_ps(dst, m_value);
    }

    inline void vector8f::store_unaligned(float* dst) const
    {
        _mm256_store_ps(dst, m_value);
    }

    inline vector8f operator-(const vector8f& rhs)
    {
        return _mm256_xor_ps(rhs, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
    }

    inline vector8f operator+(const vector8f& lhs, const vector8f& rhs)
    {
        return _mm256_add_ps(lhs, rhs);
    }

    inline vector8f operator-(const vector8f& lhs, const vector8f& rhs)
    {
        return _mm256_sub_ps(lhs, rhs);
    }

    inline vector8f operator*(const vector8f& lhs, const vector8f& rhs)
    {
        return _mm256_mul_ps(lhs, rhs);
    }

    inline vector8f operator/(const vector8f& lhs, const vector8f& rhs)
    {
        return _mm256_div_ps(lhs, rhs);
    }
    
    inline vector8fb operator==(const vector8f& lhs, const vector8f& rhs)
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_EQ_OQ);
    }

    inline vector8fb operator!=(const vector8f& lhs, const vector8f& rhs)
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NEQ_OQ);
    }

    inline vector8fb operator<(const vector8f& lhs, const vector8f& rhs)
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ);
    }

    inline vector8fb operator<=(const vector8f& lhs, const vector8f& rhs)
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_LE_OQ);
    }

    inline vector8f operator&(const vector8f& lhs, const vector8f& rhs)
    {
        return _mm256_and_ps(lhs, rhs);
    }

    inline vector8f operator|(const vector8f& lhs, const vector8f& rhs)
    {
        return _mm256_or_ps(lhs, rhs);
    }
    
    inline vector8f operator^(const vector8f& lhs, const vector8f& rhs)
    {
        return _mm256_xor_ps(lhs, rhs);
    }
    
    inline vector8f operator~(const vector8f& rhs)
    {
        return _mm256_xor_ps(rhs, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
    }

    inline vector8f min(const vector8f& lhs, const vector8f& rhs)
    {
        return _mm256_min_ps(lhs, rhs);
    }

    inline vector8f max(const vector8f& lhs, const vector8f& rhs)
    {
        return _mm256_max_ps(lhs, rhs);
    }

    inline vector8f abs(const vector8f& rhs)
    {
        __m256 sign_mask = _mm256_set1_ps(-0.f); // -0.f = 1 << 31
        return _mm256_andnot_ps(sign_mask, rhs);
    }

    inline vector8f fma(const vector8f& x, const vector8f& y, const vector8f& z)
    {
#ifdef __FMA__
        return _mm256_fmadd_ps(x, y, z);
#else
        return x * y + z;
#endif
    }

    inline vector8f sqrt(const vector8f& rhs)
    {
        return _mm256_sqrt_ps(rhs);
    }

    inline float hadd(const vector8f& rhs)
    {
        // Warning about _mm256_hadd_ps:
        // _mm256_hadd_ps(a,b) gives
        // (a0+a1,a2+a3,b0+b1,b2+b3,a4+a5,a6+a7,b4+b5,b6+b7). Hence we can't
        // rely on a naive use of this method
        // rhs = (x0, x1, x2, x3, x4, x5, x6, x7)
        // tmp = (x4, x5, x6, x7, x0, x1, x2, x3)
        __m256 tmp = _mm256_permute2f128_ps(rhs, rhs, 1);
        // tmp = (x4+x0, x5+x1, x6+x2, x7+x3, x0+x4, x1+x5, x2+x6, x3+x7)
        tmp = _mm256_add_ps(rhs, tmp);
        // tmp = (x4+x0+x5+x1, x6+x2+x7+x3, -, -, -, -, -, -)
        tmp = _mm256_hadd_ps(tmp, tmp);
        // tmp = (x4+x0+x5+x1+x6+x2+x7+x3, -, -, -, -, -, -, -)
        tmp = _mm256_hadd_ps(tmp, tmp);
        return _mm_cvtss_f32(_mm256_extractf128_ps(tmp, 0));
    }

    inline vector8f haddp(const vector8f* row)
    {
        // row = (a,b,c,d,e,f,g,h)
        // tmp0 = (a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7)
        __m256 tmp0 = _mm256_hadd_ps(row[0], row[1]);
        // tmp1 = (c0+c1, c2+c3, d1+d2, d2+d3, c4+c5, c6+c7, d4+d5, d6+d7)
        __m256 tmp1 = _mm256_hadd_ps(row[2], row[3]);
        // tmp1 = (a0+a1+a2+a3, b0+b1+b2+b3, c0+c1+c2+c3, d0+d1+d2+d3,
        // a4+a5+a6+a7, b4+b5+b6+b7, c4+c5+c6+c7, d4+d5+d6+d7)
        tmp1 = _mm256_hadd_ps(tmp0, tmp1);
        // tmp0 = (e0+e1, e2+e3, f0+f1, f2+f3, e4+e5, e6+e7, f4+f5, f6+f7)
        tmp0 = _mm256_hadd_ps(row[4], row[5]);
        // tmp2 = (g0+g1, g2+g3, h0+h1, h2+h3, g4+g5, g6+g7, h4+h5, h6+h7)
        __m256 tmp2 = _mm256_hadd_ps(row[6], row[7]);
        // tmp2 = (e0+e1+e2+e3, f0+f1+f2+f3, g0+g1+g2+g3, h0+h1+h2+h3,
        // e4+e5+e6+e7, f4+f5+f6+f7, g4+g5+g6+g7, h4+h5+h6+h7)
        tmp2 = _mm256_hadd_ps(tmp0, tmp2);
        // tmp0 = (a0+a1+a2+a3, b0+b1+b2+b3, c0+c1+c2+c3, d0+d1+d2+d3,
        // e4+e5+e6+e7, f4+f5+f6+f7, g4+g5+g6+g7, h4+h5+h6+h7)
        tmp0 = _mm256_blend_ps(tmp1, tmp2, 0b11110000);
        // tmp1 = (a4+a5+a6+a7, b4+b5+b6+b7, c4+c5+c6+c7, d4+d5+d6+d7,
        // e0+e1+e2+e3, f0+f1+f2+f3, g0+g1+g2+g3, h0+h1+h2+h3)
        tmp1 = _mm256_permute2f128_ps(tmp1, tmp2, 0x21);
        return _mm256_add_ps(tmp0, tmp1);
    }

    inline vector8f select(const vector8fb& cond, const vector8f& a, const vector8f& b)
    {
        return _mm256_blendv_ps(b, a, cond);
    }

}

#endif

