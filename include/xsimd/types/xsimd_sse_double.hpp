/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_SSE_DOUBLE_HPP
#define XSIMD_SSE_DOUBLE_HPP

#include "xsimd_base.hpp"

namespace xsimd
{

    /*************************
     * batch_bool<double, 2> *
     *************************/

    template <>
    struct simd_batch_traits<batch_bool<double, 2>>
    {
        using value_type = double;
        static constexpr std::size_t size = 2;
        using batch_type = batch<double, 2>;
        static constexpr std::size_t align = 16;
    };

    template <>
    class batch_bool<double, 2> : public simd_batch_bool<batch_bool<double, 2>>
    {
    public:

        batch_bool();
        explicit batch_bool(bool b);
        batch_bool(bool b0, bool b1);
        batch_bool(const __m128d& rhs);
        batch_bool& operator=(const __m128d& rhs);

        operator __m128d() const;

    private:

        __m128d m_value;
    };

    batch_bool<double, 2> operator&(const batch_bool<double, 2>& lhs, const batch_bool<double, 2>& rhs);
    batch_bool<double, 2> operator|(const batch_bool<double, 2>& lhs, const batch_bool<double, 2>& rhs);
    batch_bool<double, 2> operator^(const batch_bool<double, 2>& lhs, const batch_bool<double, 2>& rhs);
    batch_bool<double, 2> operator~(const batch_bool<double, 2>& rhs);
    batch_bool<double, 2> bitwise_andnot(const batch_bool<double, 2>& lhs, const batch_bool<double, 2>& rhs);

    batch_bool<double, 2> operator==(const batch_bool<double, 2>& lhs, const batch_bool<double, 2>& rhs);
    batch_bool<double, 2> operator!=(const batch_bool<double, 2>& lhs, const batch_bool<double, 2>& rhs);

    bool all(const batch_bool<double, 2>& rhs);
    bool any(const batch_bool<double, 2>& rhs);

    /********************
     * batch<double, 2> *
     ********************/

    template <>
    struct simd_batch_traits<batch<double, 2>>
    {
        using value_type = double;
        static constexpr std::size_t size = 2;
        using batch_bool_type = batch_bool<double, 2>;
        static constexpr std::size_t align = 16;
    };

    template <>
    class batch<double, 2> : public simd_batch<batch<double, 2>>
    {
    public:

        batch();
        explicit batch(double d);
        batch(double d0, double d1);
        explicit batch(const double* src);
        batch(const double* src, aligned_mode);
        batch(const double* src, unaligned_mode);
        batch(const __m128d& rhs);
        batch& operator=(const __m128d& rhs);

        operator __m128d() const;

        batch& load_aligned(const double* src);
        batch& load_unaligned(const double* src);

        batch& load_aligned(const float* src);
        batch& load_unaligned(const float* src);
        
        batch& load_aligned(const int32_t* src);
        batch& load_unaligned(const int32_t* src);
        
        batch& load_aligned(const int64_t* src);
        batch& load_unaligned(const int64_t* src);

        void store_aligned(double* dst) const;
        void store_unaligned(double* dst) const;

        void store_aligned(float* dst) const;
        void store_unaligned(float* dst) const;

        void store_aligned(int32_t* dst) const;
        void store_unaligned(int32_t* dst) const;

        void store_aligned(int64_t* dst) const;
        void store_unaligned(int64_t* dst) const;

        double operator[](std::size_t index) const;

    private:

        __m128d m_value;
    };

    batch<double, 2> operator-(const batch<double, 2>& rhs);
    batch<double, 2> operator+(const batch<double, 2>& lhs, const batch<double, 2>& rhs);
    batch<double, 2> operator-(const batch<double, 2>& lhs, const batch<double, 2>& rhs);
    batch<double, 2> operator*(const batch<double, 2>& lhs, const batch<double, 2>& rhs);
    batch<double, 2> operator/(const batch<double, 2>& lhs, const batch<double, 2>& rhs);

    batch_bool<double, 2> operator==(const batch<double, 2>& lhs, const batch<double, 2>& rhs);
    batch_bool<double, 2> operator!=(const batch<double, 2>& lhs, const batch<double, 2>& rhs);
    batch_bool<double, 2> operator<(const batch<double, 2>& lhs, const batch<double, 2>& rhs);
    batch_bool<double, 2> operator<=(const batch<double, 2>& lhs, const batch<double, 2>& rhs);

    batch<double, 2> operator&(const batch<double, 2>& lhs, const batch<double, 2>& rhs);
    batch<double, 2> operator|(const batch<double, 2>& lhs, const batch<double, 2>& rhs);
    batch<double, 2> operator^(const batch<double, 2>& lhs, const batch<double, 2>& rhs);
    batch<double, 2> operator~(const batch<double, 2>& rhs);
    batch<double, 2> bitwise_andnot(const batch<double, 2>& lhs, const batch<double, 2>& rhs);

    batch<double, 2> min(const batch<double, 2>& lhs, const batch<double, 2>& rhs);
    batch<double, 2> max(const batch<double, 2>& lhs, const batch<double, 2>& rhs);
    batch<double, 2> fmin(const batch<double, 2>& lhs, const batch<double, 2>& rhs);
    batch<double, 2> fmax(const batch<double, 2>& lhs, const batch<double, 2>& rhs);

    batch<double, 2> abs(const batch<double, 2>& rhs);
    batch<double, 2> fabs(const batch<double, 2>& rhs);
    batch<double, 2> sqrt(const batch<double, 2>& rhs);

    batch<double, 2> fma(const batch<double, 2>& x, const batch<double, 2>& y, const batch<double, 2>& z);
    batch<double, 2> fms(const batch<double, 2>& x, const batch<double, 2>& y, const batch<double, 2>& z);
    batch<double, 2> fnma(const batch<double, 2>& x, const batch<double, 2>& y, const batch<double, 2>& z);
    batch<double, 2> fnms(const batch<double, 2>& x, const batch<double, 2>& y, const batch<double, 2>& z);

    double hadd(const batch<double, 2>& rhs);
    batch<double, 2> haddp(const batch<double, 2>* row);

    batch<double, 2> select(const batch_bool<double, 2>& cond, const batch<double, 2>& a, const batch<double, 2>& b);

    batch_bool<double, 2> isnan(const batch<double, 2>& x);

    /****************************************
     * batch_bool<double, 2> implementation *
     ****************************************/

    inline batch_bool<double, 2>::batch_bool()
    {
    }

    inline batch_bool<double, 2>::batch_bool(bool b)
        : m_value(_mm_castsi128_pd(_mm_set1_epi32(-(int)b)))
    {
    }

    inline batch_bool<double, 2>::batch_bool(bool b0, bool b1)
        : m_value(_mm_castsi128_pd(_mm_setr_epi32(-(int)b0, -(int)b0, -(int)b1, -(int)b1)))
    {
    }

    inline batch_bool<double, 2>::batch_bool(const __m128d& rhs)
        : m_value(rhs)
    {
    }

    inline batch_bool<double, 2>& batch_bool<double, 2>::operator=(const __m128d& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch_bool<double, 2>::operator __m128d() const
    {
        return m_value;
    }

    inline batch_bool<double, 2> operator&(const batch_bool<double, 2>& lhs, const batch_bool<double, 2>& rhs)
    {
        return _mm_and_pd(lhs, rhs);
    }

    inline batch_bool<double, 2> operator|(const batch_bool<double, 2>& lhs, const batch_bool<double, 2>& rhs)
    {
        return _mm_or_pd(lhs, rhs);
    }

    inline batch_bool<double, 2> operator^(const batch_bool<double, 2>& lhs, const batch_bool<double, 2>& rhs)
    {
        return _mm_xor_pd(lhs, rhs);
    }

    inline batch_bool<double, 2> operator~(const batch_bool<double, 2>& rhs)
    {
        return _mm_xor_pd(rhs, _mm_castsi128_pd(_mm_set1_epi32(-1)));
    }

    inline batch_bool<double, 2> bitwise_andnot(const batch_bool<double, 2>& lhs, const batch_bool<double, 2>& rhs)
    {
        return _mm_andnot_pd(lhs, rhs);
    }

    inline batch_bool<double, 2> operator==(const batch_bool<double, 2>& lhs, const batch_bool<double, 2>& rhs)
    {
        return _mm_cmpeq_pd(lhs, rhs);
    }

    inline batch_bool<double, 2> operator!=(const batch_bool<double, 2>& lhs, const batch_bool<double, 2>& rhs)
    {
        return _mm_cmpneq_pd(lhs, rhs);
    }

    inline bool all(const batch_bool<double, 2>& rhs)
    {
        return _mm_movemask_pd(rhs) == 3;
    }

    inline bool any(const batch_bool<double, 2>& rhs)
    {
        return _mm_movemask_pd(rhs) != 0;
    }

    /***********************************
     * batch<double, 2> implementation *
     ***********************************/

    inline batch<double, 2>::batch()
    {
    }

    inline batch<double, 2>::batch(double d)
        : m_value(_mm_set1_pd(d))
    {
    }

    inline batch<double, 2>::batch(double d0, double d1)
        : m_value(_mm_setr_pd(d0, d1))
    {
    }

    inline batch<double, 2>::batch(const double* src)
        : m_value(_mm_loadu_pd(src))
    {
    }

    inline batch<double, 2>::batch(const double* src, aligned_mode)
        : m_value(_mm_load_pd(src))
    {
    }

    inline batch<double, 2>::batch(const double* src, unaligned_mode)
        : m_value(_mm_loadu_pd(src))
    {
    }

    inline batch<double, 2>::batch(const __m128d& rhs)
        : m_value(rhs)
    {
    }

    inline batch<double, 2>& batch<double, 2>::operator=(const __m128d& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<double, 2>::operator __m128d() const
    {
        return m_value;
    }

    inline batch<double, 2>& batch<double, 2>::load_aligned(const double* src)
    {
        m_value = _mm_load_pd(src);
        return *this;
    }

    inline batch<double, 2>& batch<double, 2>::load_unaligned(const double* src)
    {
        m_value = _mm_loadu_pd(src);
        return *this;
    }

    inline batch<double, 2>& batch<double, 2>::load_aligned(const float* src)
    {
        alignas(16) double tmp[2];
        tmp[0] = double(src[0]);
        tmp[1] = double(src[1]);
        m_value = _mm_load_pd(tmp);
        return *this;
    }

    inline batch<double, 2>& batch<double, 2>::load_unaligned(const float* src)
    {
        return load_aligned(src);
    }

    inline batch<double, 2>& batch<double, 2>::load_aligned(const int32_t* src)
    {
        alignas(16) double tmp[2];
        tmp[0] = double(src[0]);
        tmp[1] = double(src[1]);
        m_value = _mm_load_pd(tmp);
        return *this;
    }

    inline batch<double, 2>& batch<double, 2>::load_unaligned(const int32_t* src)
    {
        return load_aligned(src);
    }

    inline batch<double, 2>& batch<double, 2>::load_aligned(const int64_t* src)
    {
        alignas(16) double tmp[2];
        tmp[0] = double(src[0]);
        tmp[1] = double(src[1]);
        m_value = _mm_load_pd(tmp);
        return *this;
    }

    inline batch<double, 2>& batch<double, 2>::load_unaligned(const int64_t* src)
    {
        return load_aligned(src);
    }

    inline void batch<double, 2>::store_aligned(double* dst) const
    {
        _mm_store_pd(dst, m_value);
    }

    inline void batch<double, 2>::store_unaligned(double* dst) const
    {
        _mm_storeu_pd(dst, m_value);
    }

    inline void batch<double, 2>::store_aligned(float* dst) const
    {
        alignas(16) double tmp[2];
        _mm_store_pd(tmp, m_value);
        dst[0] = static_cast<float>(tmp[0]);
        dst[1] = static_cast<float>(tmp[1]);
    }

    inline void batch<double, 2>::store_unaligned(float* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<double, 2>::store_aligned(int32_t* dst) const
    {
        alignas(16) double tmp[2];
        _mm_store_pd(tmp, m_value);
        dst[0] = static_cast<int32_t>(tmp[0]);
        dst[1] = static_cast<int32_t>(tmp[1]);
    }

    inline void batch<double, 2>::store_unaligned(int32_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<double, 2>::store_aligned(int64_t* dst) const
    {
        alignas(16) double tmp[2];
        _mm_store_pd(tmp, m_value);
        dst[0] = static_cast<int64_t>(tmp[0]);
        dst[1] = static_cast<int64_t>(tmp[1]);
    }

    inline void batch<double, 2>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline double batch<double, 2>::operator[](std::size_t index) const
    {
        alignas(16) double x[2];
        store_aligned(x);
        return x[index & 1];
    }

    inline batch<double, 2> operator-(const batch<double, 2>& rhs)
    {
        return _mm_xor_pd(rhs, _mm_castsi128_pd(_mm_setr_epi32(0, 0x80000000,
                                                               0, 0x80000000)));
    }

    inline batch<double, 2> operator+(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_add_pd(lhs, rhs);
    }

    inline batch<double, 2> operator-(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_sub_pd(lhs, rhs);
    }

    inline batch<double, 2> operator*(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_mul_pd(lhs, rhs);
    }

    inline batch<double, 2> operator/(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_div_pd(lhs, rhs);
    }

    inline batch_bool<double, 2> operator==(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_cmpeq_pd(lhs, rhs);
    }

    inline batch_bool<double, 2> operator!=(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_cmpneq_pd(lhs, rhs);
    }

    inline batch_bool<double, 2> operator<(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_cmplt_pd(lhs, rhs);
    }

    inline batch_bool<double, 2> operator<=(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_cmple_pd(lhs, rhs);
    }

    inline batch<double, 2> operator&(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_and_pd(lhs, rhs);
    }

    inline batch<double, 2> operator|(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_or_pd(lhs, rhs);
    }

    inline batch<double, 2> operator^(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_xor_pd(lhs, rhs);
    }

    inline batch<double, 2> operator~(const batch<double, 2>& rhs)
    {
        return _mm_xor_pd(rhs, _mm_castsi128_pd(_mm_set1_epi32(-1)));
    }

    inline batch<double, 2> bitwise_andnot(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_andnot_pd(lhs, rhs);
    }

    inline batch<double, 2> min(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_min_pd(lhs, rhs);
    }

    inline batch<double, 2> max(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return _mm_max_pd(lhs, rhs);
    }

    inline batch<double, 2> fmin(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return min(lhs, rhs);
    }

    inline batch<double, 2> fmax(const batch<double, 2>& lhs, const batch<double, 2>& rhs)
    {
        return max(lhs, rhs);
    }

    inline batch<double, 2> abs(const batch<double, 2>& rhs)
    {
        __m128d sign_mask = _mm_set1_pd(-0.);  // -0. = 1 << 63
        return _mm_andnot_pd(sign_mask, rhs);
    }

    inline batch<double, 2> fabs(const batch<double, 2>& rhs)
    {
        return abs(rhs);
    }

    inline batch<double, 2> sqrt(const batch<double, 2>& rhs)
    {
        return _mm_sqrt_pd(rhs);
    }

    inline batch<double, 2> fma(const batch<double, 2>& x, const batch<double, 2>& y, const batch<double, 2>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm_fmadd_pd(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm_macc_pd(x, y, z);
#else
        return x * y + z;
#endif
    }

    inline batch<double, 2> fms(const batch<double, 2>& x, const batch<double, 2>& y, const batch<double, 2>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm_fmsub_pd(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm_msub_pd(x, y, z);
#else
        return x * y - z;
#endif
    }

    inline batch<double, 2> fnma(const batch<double, 2>& x, const batch<double, 2>& y, const batch<double, 2>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm_fnmadd_pd(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm_nmacc_pd(x, y, z);
#else
        return -x * y + z;
#endif
    }

    inline batch<double, 2> fnms(const batch<double, 2>& x, const batch<double, 2>& y, const batch<double, 2>& z)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_FMA3_VERSION
        return _mm_fnmsub_pd(x, y, z);
#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AMD_FMA4_VERSION
        return _mm_nmsub_pd(x, y, z);
#else
        return -x * y - z;
#endif
    }

    inline double hadd(const batch<double, 2>& rhs)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE3_VERSION
        __m128d tmp0 = _mm_hadd_pd(rhs, rhs);
#else
        __m128d tmp0 = _mm_add_sd(rhs, _mm_unpackhi_pd(rhs, rhs));
#endif
        return _mm_cvtsd_f64(tmp0);
    }

    inline batch<double, 2> haddp(const batch<double, 2>* row)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE3_VERSION
        return _mm_hadd_pd(row[0], row[1]);
#else
        return _mm_add_pd(_mm_unpacklo_pd(row[0], row[1]),
                          _mm_unpackhi_pd(row[0], row[1]));
#endif
    }

    inline batch<double, 2> select(const batch_bool<double, 2>& cond, const batch<double, 2>& a, const batch<double, 2>& b)
    {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        return _mm_blendv_pd(b, a, cond);
#else
        return _mm_or_pd(_mm_and_pd(cond, a), _mm_andnot_pd(cond, b));
#endif
    }

    inline batch_bool<double, 2> isnan(const batch<double, 2>& x)
    {
        return _mm_cmpunord_pd(x, x);
    }
}

#endif
