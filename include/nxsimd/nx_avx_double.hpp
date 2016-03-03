//
// Copyright (c) 2012 - 2016 Johan Mabille
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
//

#ifndef NX_AVX_DOUBLE_HPP
#define NX_AVX_DOUBLE_HPP

#include "nx_simd_base.hpp"

namespace nxsimd
{

    class vector4db : public simd_vector_bool<vector4db>
    {

    public:

        vector4db();
        vector4db(bool b);
        vector4db(bool b0, bool b1, bool b2, bool b3);
        vector4db(const __m256& rhs);
        vector4db& operator=(const __m256& rhs);

        operator __256() const;

    private:

        __m256 m_value;
    };

    vector4db operator&(const vector4db& lhs, const vector4db& rhs);
    vector4db operator|(const vector4db& lhs, const vector4db& rhs);
    vector4db operator^(const vector4db& lhs, const vector4db& rhs);
    vector4db operator~(const vector4db& rhs);

    vector4db operator==(const vector4db& lhs, const vector4db& rhs);
    vector4db operator!=(const vector4db& lhs, const vector4db& rhs);

    class vector4d;

    template <>
    struct simd_vector_traits<vector4d>
    {
        using value_type = float;
        using vector_bool = vector4db;
    };

    class vector4d : public simd_vector<vector4d>
    {

    public:

        vector4d();
        vector4d(double d);
        vector4d(double d0, double d1, double d2, double d3);
        vector4d(const __m128& rhs);
        vector4d& operator=(const __m128& rhs);

        operator __m128() const;

        vector4d& load_aligned(const double* src);
        vector4d& load_unaligned(const double* src);

        void store_aligned(double* dst) const;
        void store_unaligned(double* dst) const;

    private:

        __m128 m_value;
    };

    vector4d operator-(const vector4d& rhs);
    vector4d operator+(const vector4d& lhs, const vector4d& rhs);
    vector4d operator-(const vector4d& lhs, const vector4d& rhs);
    vector4d operator*(const vector4d& lhs, const vector4d& rhs);
    vector4d operator/(const vector4d& lhs, const vector4d& rhs);
    
    vector4db operator==(const vector4d& lhs, const vector4d& rhs);
    vector4db operator!=(const vector4d& lhs, const vector4d& rhs);
    vector4db operator<(const vector4d& lhs, const vector4d& rhs);
    vector4db operator<=(const vector4d& lhs, const vector4d& rhs);

    vector4d operator&(const vector4d& lhs, const vector4d& rhs);
    vector4d operator|(const vector4d& lhs, const vector4d& rhs);
    vector4d operator^(const vector4d& lhs, const vector4d& rhs);
    vector4d operator~(const vector4d& rhs);

    vector4d min(const vector4d& lhs, const vector4d& rhs);
    vector4d max(const vector4d& lhs, const vector4d& rhs);

    vector4d abs(const vector4d& rhs);

    vector4d fma(const vector4d& x, const vector4d& y, const vector4d& z);

    vector4d sqrt(const vector4d& rhs);

    double hadd(const vector4d& rhs);
    vector4d haddp(const vector4d* row);

    vector4d select(const vector4db& cond, const vector4d& a, const vector4d& b);


    /******************************
     * vector4db implementation
     ******************************/

    inline vector4db::vector4db()
    {
    }

    inline vector4db(bool b)
        : m_value(_mm256_castsi256_pd(_mm256_set1_epi32(-(int)b)))
    {
    }

    inline vector4db::vector4db(bool b0, bool b1, bool b2, bool b3)
        : m_value(_mm256_castsi256_pd(
                  _mm256_setr_epi32(-(int)b0, -(int)b0, -(int)b1, -(int)b1,
                                    -(int)b2, -(int)b2, -(int)b3, -(int)b3)))
    {
    }

    inline vector4db::vector4db(const __m256& rhs)
        : m_value(rhs)
    {
    }

    inline vector4db& vector4db::operator=(const __m256& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline vector4db::operator __256() const
    {
        return *this;
    }

    inline vector4db operator&(const vector4db& lhs, const vector4db& rhs)
    {
        return _mm256_and_pd(lhs, rhs);
    }

    inline vector4db operator|(const vector4db& lhs, const vector4db& rhs)
    {
        return _mm256_or_pd(lhs, rhs);
    }

    inline vector4db operator^(const vector4db& lhs, const vector4db& rhs)
    {
        return _mm256_xor_pd(lhs, rhs);
    }

    inline vector4db operator~(const vector4db& rhs)
    {
        return _mm256_xor_pd(rhs, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
    }

    inline vector4db operator==(const vector4db& lhs, const vector4db& rhs)
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_EQ_OQ);
    }

    inline vector4db operator!=(const vector4db& lhs, const vector4db& rhs)
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_NEQ_OQ);
    }


    /*****************************
     * vector4d implementation
     *****************************/

    inline vector4d::vector4d()
    {
    }

    inline vector4d::vector4d(double d)
        : m_value(_mm256_set1_pd(d))
    {
    }

    inline vector4d::vector4d(double d0, double d1, double d2, double d3)
        : m_value(_mm256_setr_pd(d0, d1, d2, d3))
    {
    }
    
    inline vector4d::vector4d(const __m128& rhs)
        : m_value(rhs)
    {
    }

    inline vector4d& vector4d::operator=(const __m128& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline vector4d::operator __m128() const
    {
        return m_value;
    }

    inline vector4d& vector4d::load_aligned(const double* src)
    {
        m_value = _mm256_load_pd(src);
        return *this;
    }

    inline vector4d& vector4d::load_unaligned(const double* src)
    {
        m_value = _mm256_loadu_pd(src);
        return *this;
    }

    inline void vector4d::store_aligned(double* dst) const
    {
        _mm256_store_pd(dst, m_value);
    }

    inline void vector4d::store_unaligned(double* dst) const
    {
        _mm256_storeu_pd(dst, m_value);
    }

    inline vector4d operator-(const vector4d& rhs)
    {
        return _mm256_xor_pd(rhs, _mm256_castsi256_pd(_mm256_set1_epi32(0x80000000)));
    }

    inline vector4d operator+(const vector4d& lhs, const vector4d& rhs)
    {
        return _mm256_add_pd(lhs, rhs);
    }

    inline vector4d operator-(const vector4d& lhs, const vector4d& rhs)
    {
        return _mm256_sub_pd(lhs, rhs);
    }

    inline vector4d operator*(const vector4d& lhs, const vector4d& rhs)
    {
        return _mm256_mul_pd(lhs, rhs);
    }

    inline vector4d operator/(const vector4d& lhs, const vector4d& rhs)
    {
        return _mm256_div_pd(lhs, rhs);
    }
    
    inline vector4db operator==(const vector4d& lhs, const vector4d& rhs)
    {
        return _mm256_cmp_pd(lhs, rhs _CMP_EQ_OQ);
    }

    inline vector4db operator!=(const vector4d& lhs, const vector4d& rhs)
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_NEQ_OQ);
    }

    inline vector4db operator<(const vector4d& lhs, const vector4d& rhs)
    {
        return _mm256_cmp(lhs, rhs, _CMP_LT_OQ);
    }

    inline vector4db operator<=(const vector4d& lhs, const vector4d& rhs)
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_LE_OQ);
    }

    inline vector4d operator&(const vector4d& lhs, const vector4d& rhs)
    {
        return _mm256_and_pd(lhs, rhs);
    }

    inline vector4d operator|(const vector4d& lhs, const vector4d& rhs)
    {
        return _mm256_or_pd(lhs, rhs);
    }

    inline vector4d operator^(const vector4d& lhs, const vector4d& rhs)
    {
        return _mm256_xor_pd(lhs, rhs);
    }

    inline vector4d operator~(const vector4d& rhs)
    {
        return _mm256_xor_pd(rhs, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
    }

    inline vector4d min(const vector4d& lhs, const vector4d& rhs)
    {
        return _mm256_min_pd(lhs, rhs);
    }

    inline vector4d max(const vector4d& lhs, const vector4d& rhs)
    {
        return _mm256_max_pd(lhs, rhs);
    }

    inline vector4d abs(const vector4d& rhs)
    {
        __m256d sign_mask = _mm256_set1_pd(-0.); // -0. = 1 << 63
        return _mm256_andnot_pd(sign_mask, rhs);
    }

    inline vector4d fma(const vector4d& x, const vector4d& y, const vector4d& z)
    {
#ifdef __FMA__
        return _mm256_fmadd_pd(x, y, z);
#else
        return x * y + z;
#endif
    }

    inline vector4d sqrt(const vector4d& rhs)
    {
        return _mm256_sqrt_pd(rhs);
    }

    double hadd(const vector4d& rhs)
    {
        // rhs = (x0, x1, x2, x3)
        // tmp = (x2, x3, x0, x1)
        __m256d tmp = _mm256_permute2f128_pd(rhs, rhs, 1);
        // tmp = (x2+x0, x3+x1, -, -)
        tmp = _mm256_add_pd(rhs, tmp);
        // tmp = (x2+x0+x3+x1, -, -, -)
        tmp = _mm256_hadd_pd(tmp, tmp);
        return _mm_cvtsd_f64(_mm256_extractf128_pd(tmp, 0));
    }

    inline vector4d haddp(const vector4d* row)
    {
        // row = (a,b,c,d)
        // tmp0 = (a0+a1, b0+b1, a2+a3, b2+b3)
        __m256d tmp0 = _mm256_hadd_pd(row[0], row[1]);
        // tmp1 = (c0+c1, d0+d1, c2+c3, d2+d3)
        __m256d tmp1 = _mm256_hadd_pd(row[2], row[3]);
        // tmp2 = (a0+a1, b0+b1, c2+c3, d2+d3)
        __m256d tmp2 = _mm256_blend_pd(tmp0, tmp1, 0b1100);
        // tmp1 = (a2+a3, b2+b3, c2+c3, d2+d3)
        tmp1 = _mm256_permute2f128_pd(tmp0, tmp1, 0x21);
        return _mm256_add_pd(tmp1, tmp2);
    }

    inline vector4d select(const vector4db& cond, const vector4d& a, const vector4d& b)
    {
        return _mm256_blendv_pd(b, a, cond);
    }

}

#endif
