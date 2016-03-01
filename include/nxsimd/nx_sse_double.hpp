#ifndef NX_SSE_DOUBLE_HPP
#define NX_SSE_DOUBLE_HPP

namespace nxsimd
{

    class vector2db : public simd_vector_bool<vector2db>
    {

    public:

        vector2db();
        vector2db(bool b);
        vector2db(bool b0, bool b1);
        vector2db(const __m128d& rhs);
        vector2db& operator=(const __m128d& rhs);

        operator __m128d() const;

    private:

        __m128d m_value;
    };

    vector2db operator&(const vector2db& lhs, const vector2db& rhs);
    vector2db operator|(const vector2db& lhs, const vector2db& rhs);
    vector2db operator^(const vector2db& lhs, const vector2db& rhs);
    vector2db operator~(const vector2db& rhs);

    vector2db operator==(const vector2db& lhs, const vector2db& rhs);
    vector2db operator!=(const vector2db& lhs, const vector2db& rhs);

    class vector2d;

    template <>
    struct simd_vector_traits<vector2d>
    {
        using value_type = double;
        using vector_bool = vector2db;
    };

    class vector2d : public simd_vector<vector2d>
    {

    public:

        vector2d();
        vector2d(double d);
        vector2d(double d0, double d1);
        vector2d(const __m128d& rhs);
        vector2d& operator=(const __m128d& rhs);

        operator __m128d() const;

        vector2d& load_aligned(const double* src);
        vector2d& load_unaligned(const double* src);

        void store_aligned(double* dst);
        void store_unaligned(double* dst);

    private:

        __m128d m_value;
    };

    vector2d operator-(const vector2d& rhs);
    vector2d operator+(const vector2d& lhs, const vector2d& rhs);
    vector2d operator-(const vector2d& lhs, const vector2d& rhs);
    vector2d operator*(const vector2d& lhs, const vector2d& rhs);
    vector2d operator/(const vector2d& lhs, const vector2d& rhs);
    
    vector2db operator==(const vector2d& lhs, const vector2d& rhs);
    vector2db operator!=(const vector2d& lhs, const vector2d& rhs);
    vector2db operator<(const vector2d& lhs, const vector2d& rhs);
    vector2db operator<=(const vector2d& lhs, const vector2d& rhs);

    vector2d operator&(const vector2d& lhs, const vector2d& rhs);
    vector2d operator|(const vector2d& lhs, const vector2d& rhs);
    vector2d operator^(const vector2d& lhs, const vector2d& rhs);
    vector2d operator~(const vector2d& rhs);

    double hadd(const vector2d& rhs);
    vector2d haddp(const vector2d* row);

    vector2d select(const vector2db& cond, const vector2d& a, const vector2d& b);


    /******************************
     * vector2db implementation
     ******************************/

    inline vector2db::vector2db()
    {
    }

    inline vector2db::vector2db(bool b)
        : m_value(_mm_castsi128_pd(_mm_set_epi32(-(int)b)))
    {
    }

    inline vector2db::vector2db(bool b0, bool b1)
        : m_value(_mm_castsi128_pd(_mm_setr_epi32(-(int)b0, -(int)b1, -(int)b2, -(int)b3)))
    {
    }

    inline vector2db::vector2db(const __m128d& rhs)
        : m_value(rhs)
    {
    }

    inline vector2db::vector2db& operator=(const __m128d& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline vector2db::operator __m128d() const
    {
        return m_value;
    }

    inline vector2db operator&(const vector2db& lhs, const vector2db& rhs)
    {
        return _mm_and_pd(lhs, rhs);
    }

    inline vector2db operator|(const vector2db& lhs, const vector2db& rhs)
    {
        return _mm_or_pd(lhs, rhs);
    }

    inline vector2db operator^(const vector2db& lhs, const vector2db& rhs)
    {
        return _mm_xor_pd(lhs, rhs);
    }

    inline vector2db operator~(const vector2db& rhs)
    {
        return _mm_xor_pd(rhs, _mm_castsi128_pd(_mm_set_epi32(-1)));
    }

    inline vector2db operator==(const vector2db& lhs, const vector2db& rhs)
    {
        return _mm_cmpeq_pd(lhs, rgs);
    }

    inline vector2db operator!=(const vector2db& lhs, const vector2db& rhs)
    {
        return _mm_cmpneq_pd(lhs, rhs);
    }


    /*****************************
     * vector2d implementation
     *****************************/

    inline vector2d::vector2d()
    {
    }

    inline vector2d::vector2d(double d)
        : m_value(_mm_set1_pd(d))
    {
    }

    inline vector2d::vector2d(double d0, double d1)
        : m_value(_mm_setr_pd(d0, d1))
    {
    }

    inline vector2d::vector2d(const __m128d& rhs)
        : m_value(rhs)
    {
    }

    inline vector2d::vector2d& operator=(const __m128d& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline vector2d::operator __m128d() const
    {
        return m_value;
    }

    inline vector2d& vector2d::load_aligned(const double* src)
    {
        m_value = _mm_load_pd(src);
        return *this;
    }

    inline vector2d& vector2d::load_unaligned(const double* src)
    {
        m_value = _mm_loadu_pd(src);
        return *this;
    }

    inline void vector2d::store_aligned(double* dst)
    {
        _mm_store_pd(dst, m_value);
    }

    inline void vector2d::store_unaligned(double* dst)
    {
        _mm_storu_pd(dst, value);
    }

    inline vector2d operator-(const vector2d& rhs)
    {
        return _mm_xor_pd(_mm_castsi128_pd(_mm_setr_epi32(0, 0x80000000,
                                                          0, 0x80000000)));
    }

    inline vector2d operator+(const vector2d& lhs, const vector2d& rhs)
    {
        return _mm_add_pd(lhs, rhs);
    }

    inline vector2d operator-(const vector2d& lhs, const vector2d& rhs)
    {
        return _mm_sub_pd(lhs, rhs);
    }

    inline vector2d operator*(const vector2d& lhs, const vector2d& rhs)
    {
        return _mm_mul_pd(lhs, rhs);
    }

    inline vector2d operator/(const vector2d& lhs, const vector2d& rhs)
    {
        return _mm_div_pd(lhs, rhs);
    }
    
    inline vector2db operator==(const vector2d& lhs, const vector2d& rhs)
    {
        return _mm_cmpeq_pd(lhs, rhs);
    }

    inline vector2db operator!=(const vector2d& lhs, const vector2d& rhs)
    {
        return _mm_cmpneq(lhs, rhs);
    }

    inline vector2db operator<(const vector2d& lhs, const vector2d& rhs)
    {
        return _mm_cmplt_pd(lhs, rhs);
    }

    inline vector2db operator<=(const vector2d& lhs, const vector2d& rhs)
    {
        return _mm_cmple_pd(lhs, rhs);
    }

    inline vector2d operator&(const vector2d& lhs, const vector2d& rhs)
    {
        return _mm_and_pd(lhs, rhs);
    }

    inline vector2d operator|(const vector2d& lhs, const vector2d& rhs)
    {
        return _mm_or_pd(lhs, rhs);
    }

    inline vector2d operator^(const vector2d& lhs, const vector2d& rhs)
    {
        return _mm_xor_pd(lhs, rhs);
    }

    inline vector2d operator~(const vector2d& rhs)
    {
        return _mm_xor_pd(rhs, _mm_castsi128_pd(_mm_set1_epi32(-1)));
    }

    inline double hadd(const vector2d& rhs)
    {
#if SSE_INSTR_SET >= 3  // SSE3
        __m128d tmp0 = _mm_hadd_pd(rhs, rhs);
#else
        __m128d tmp0 = _mm_add_sd(rhs, _mm_unpackhi_pd(rhs, rhs));
#endif
        return _mm_cvtsd_f64(tmp0);
    }

    inline vector2d haddp(const vector2d* row)
    {
#if SSE_INSTR_SET >= 3  // SSE3
        return _mm_hadd_pd(row[0], row[1]);
#else
        return _mm_add_pd(_mm_unpacklo_pd(row[0], row[1]),
                          _mm_unpackhi_pd(row[0], row[1]));
#endif
    }

    inline vector2d select(const vector2db& cond, const vector2d& a, const vector2d& b)
    {
#if SSE_INSTR_SET >= 5  // SSE 4.1
        return _mm_blendv_pd(b, a, cond);
#else
        return _mm_or_pd(_mm_and_pd(cond, a), _mm_andnot_pd(cond, b));
#endif
    }

}

#endif

