/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_DOUBLE_HPP
#define XSIMD_AVX512_DOUBLE_HPP

#include "xsimd_avx512_bool.hpp"
#include "xsimd_base.hpp"

namespace xsimd
{

    /*************************
     * batch_bool<double, 8> *
     *************************/

    template <>
    struct simd_batch_traits<batch_bool<double, 8>>
    {
        using value_type = double;
        static constexpr std::size_t size = 8;
        using batch_type = batch<double, 8>;
    };

    template <>
    class batch_bool<double, 8> : 
        public batch_bool_avx512<__mmask8, batch_bool<double, 8>>,
        public simd_batch_bool<batch_bool<double, 8>>
    {
    public:
        using base_class = batch_bool_avx512<__mmask8, batch_bool<double, 8>>;
        using base_class::base_class;

        batch_bool(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7)
            : base_class({{b0, b1, b2, b3, b4, b5, b6, b7}})
        {
        }
    };

    namespace detail
    {
        template <>
        struct batch_bool_kernel<double, 8>
            : batch_bool_kernel_avx512<double, 8>
        {
        };
    }

    /********************
     * batch<double, 8> *
     ********************/

    template <>
    struct simd_batch_traits<batch<double, 8>>
    {
        using value_type = double;
        static constexpr std::size_t size = 8;
        using batch_bool_type = batch_bool<double, 8>;
    };

    template <>
    class batch<double, 8> : public simd_batch<batch<double, 8>>
    {
    public:

        batch();
        explicit batch(double d);
        batch(double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7);
        explicit batch(const double* src);
        batch(const double* src, aligned_mode);
        batch(const double* src, unaligned_mode);
        batch(const __m512d& rhs);
        batch& operator=(const __m512d& rhs);

        operator __m512d() const;

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

        __m512d m_value;
    };

    batch<double, 8> operator-(const batch<double, 8>& rhs);
    batch<double, 8> operator+(const batch<double, 8>& lhs, const batch<double, 8>& rhs);
    batch<double, 8> operator-(const batch<double, 8>& lhs, const batch<double, 8>& rhs);
    batch<double, 8> operator*(const batch<double, 8>& lhs, const batch<double, 8>& rhs);
    batch<double, 8> operator/(const batch<double, 8>& lhs, const batch<double, 8>& rhs);

    batch_bool<double, 8> operator==(const batch<double, 8>& lhs, const batch<double, 8>& rhs);
    batch_bool<double, 8> operator!=(const batch<double, 8>& lhs, const batch<double, 8>& rhs);
    batch_bool<double, 8> operator<(const batch<double, 8>& lhs, const batch<double, 8>& rhs);
    batch_bool<double, 8> operator<=(const batch<double, 8>& lhs, const batch<double, 8>& rhs);

    batch<double, 8> operator&(const batch<double, 8>& lhs, const batch<double, 8>& rhs);
    batch<double, 8> operator|(const batch<double, 8>& lhs, const batch<double, 8>& rhs);
    batch<double, 8> operator^(const batch<double, 8>& lhs, const batch<double, 8>& rhs);
    batch<double, 8> operator~(const batch<double, 8>& rhs);
    batch<double, 8> bitwise_andnot(const batch<double, 8>& lhs, const batch<double, 8>& rhs);

    batch<double, 8> min(const batch<double, 8>& lhs, const batch<double, 8>& rhs);
    batch<double, 8> max(const batch<double, 8>& lhs, const batch<double, 8>& rhs);
    batch<double, 8> fmin(const batch<double, 8>& lhs, const batch<double, 8>& rhs);
    batch<double, 8> fmax(const batch<double, 8>& lhs, const batch<double, 8>& rhs);

    batch<double, 8> abs(const batch<double, 8>& rhs);
    batch<double, 8> fabs(const batch<double, 8>& rhs);
    batch<double, 8> sqrt(const batch<double, 8>& rhs);

    batch<double, 8> fma(const batch<double, 8>& x, const batch<double, 8>& y, const batch<double, 8>& z);
    batch<double, 8> fms(const batch<double, 8>& x, const batch<double, 8>& y, const batch<double, 8>& z);
    batch<double, 8> fnma(const batch<double, 8>& x, const batch<double, 8>& y, const batch<double, 8>& z);
    batch<double, 8> fnms(const batch<double, 8>& x, const batch<double, 8>& y, const batch<double, 8>& z);

    double hadd(const batch<double, 8>& rhs);
    batch<double, 8> haddp(const batch<double, 8>* row);

    batch<double, 8> select(const batch_bool<double, 8>& cond, const batch<double, 8>& a, const batch<double, 8>& b);

    batch_bool<double, 8> isnan(const batch<double, 8>& x);

    /***********************************
     * batch<double, 8> implementation *
     ***********************************/

    inline batch<double, 8>::batch()
    {
    }

    inline batch<double, 8>::batch(double d)
        : m_value(_mm512_set1_pd(d))
    {
    }

    inline batch<double, 8>::batch(double d0, double d1, double d2, double d3, double d4, double d5, double d6, double d7)
        : m_value(_mm512_setr_pd(d0, d1, d2, d3, d4, d5, d6, d7))
    {
    }

    inline batch<double, 8>::batch(const double* src)
        : m_value(_mm512_loadu_pd(src))
    {
    }

    inline batch<double, 8>::batch(const double* src, aligned_mode)
        : m_value(_mm512_load_pd(src))
    {
    }

    inline batch<double, 8>::batch(const double* src, unaligned_mode)
        : m_value(_mm512_loadu_pd(src))
    {
    }

    inline batch<double, 8>::batch(const __m512d& rhs)
        : m_value(rhs)
    {
    }

    inline batch<double, 8>& batch<double, 8>::operator=(const __m512d& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<double, 8>::operator __m512d() const
    {
        return m_value;
    }

    inline batch<double, 8>& batch<double, 8>::load_aligned(const double* src)
    {
        m_value = _mm512_load_pd(src);
        return *this;
    }

    inline batch<double, 8>& batch<double, 8>::load_unaligned(const double* src)
    {
        m_value = _mm512_loadu_pd(src);
        return *this;
    }

    inline batch<double, 8>& batch<double, 8>::load_aligned(const float* src)
    {
        m_value = _mm512_cvtps_pd(_mm256_load_ps(src));
        return *this;
    }

    inline batch<double, 8>& batch<double, 8>::load_unaligned(const float* src)
    {
        m_value = _mm512_cvtps_pd(_mm256_loadu_ps(src));
        return *this;
    }

    inline batch<double, 8>& batch<double, 8>::load_aligned(const int32_t* src)
    {
        m_value = _mm512_cvtepi32_pd(_mm256_load_si256((__m256i const*)src));
        return *this;
    }

    inline batch<double, 8>& batch<double, 8>::load_unaligned(const int32_t* src)
    {
        m_value = _mm512_cvtepi32_pd(_mm256_loadu_si256((__m256i const*)src));
        return *this;
    }

    inline batch<double, 8>& batch<double, 8>::load_aligned(const int64_t* src)
    {
        alignas(64) double tmp[8];
        tmp[0] = double(src[0]);
        tmp[1] = double(src[1]);
        tmp[2] = double(src[2]);
        tmp[3] = double(src[3]);
        tmp[4] = double(src[4]);
        tmp[5] = double(src[5]);
        tmp[6] = double(src[6]);
        tmp[7] = double(src[7]);
        m_value = _mm512_load_pd(tmp);
        return *this;
    }

    inline batch<double, 8>& batch<double, 8>::load_unaligned(const int64_t* src)
    {
        return load_aligned(src);
    }

    inline void batch<double, 8>::store_aligned(double* dst) const
    {
        _mm512_store_pd(dst, m_value);
    }

    inline void batch<double, 8>::store_unaligned(double* dst) const
    {
        _mm512_storeu_pd(dst, m_value);
    }

    inline void batch<double, 8>::store_aligned(float* dst) const
    {
        _mm256_store_ps(dst, _mm512_cvtpd_ps(m_value));
    }

    inline void batch<double, 8>::store_unaligned(float* dst) const
    {
        _mm256_storeu_ps(dst, _mm512_cvtpd_ps(m_value));
    }

    inline void batch<double, 8>::store_aligned(int32_t* dst) const
    {
        _mm256_store_si256((__m256i*)dst, _mm512_cvtpd_epi32(m_value));
    }

    inline void batch<double, 8>::store_unaligned(int32_t* dst) const
    {
        _mm256_storeu_si256((__m256i*)dst, _mm512_cvtpd_epi32(m_value));
    }

    inline void batch<double, 8>::store_aligned(int64_t* dst) const
    {
        // TODO check if intrinsic available
        alignas(64) double tmp[8];
        _mm512_store_pd(tmp, m_value);
        dst[0] = static_cast<int64_t>(tmp[0]);
        dst[1] = static_cast<int64_t>(tmp[1]);
        dst[2] = static_cast<int64_t>(tmp[2]);
        dst[3] = static_cast<int64_t>(tmp[3]);
        dst[4] = static_cast<int64_t>(tmp[4]);
        dst[5] = static_cast<int64_t>(tmp[5]);
        dst[6] = static_cast<int64_t>(tmp[6]);
        dst[7] = static_cast<int64_t>(tmp[7]);
    }

    inline void batch<double, 8>::store_unaligned(int64_t* dst) const
    {
        store_aligned(dst);
    }

    inline double batch<double, 8>::operator[](std::size_t index) const
    {
        alignas(64) double x[8];
        store_aligned(x);
        return x[index & 7];
    }

    inline batch<double, 8> operator-(const batch<double, 8>& rhs)
    {
        return _mm512_xor_pd(rhs, _mm512_castsi512_pd(_mm512_set1_epi64(0x8000000000000000)));
    }

    inline batch<double, 8> operator+(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_add_pd(lhs, rhs);
    }

    inline batch<double, 8> operator-(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_sub_pd(lhs, rhs);
    }

    inline batch<double, 8> operator*(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_mul_pd(lhs, rhs);
    }

    inline batch<double, 8> operator/(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_div_pd(lhs, rhs);
    }

    inline batch_bool<double, 8> operator==(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_cmp_pd_mask(lhs, rhs, _CMP_EQ_OQ);
    }

    inline batch_bool<double, 8> operator!=(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_cmp_pd_mask(lhs, rhs, _CMP_NEQ_OQ);
    }

    inline batch_bool<double, 8> operator<(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_cmp_pd_mask(lhs, rhs, _CMP_LT_OQ);
    }

    inline batch_bool<double, 8> operator<=(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_cmp_pd_mask(lhs, rhs, _CMP_LE_OQ);
    }

    inline batch<double, 8> operator&(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_and_pd(lhs, rhs);
    }

    inline batch<double, 8> operator|(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_or_pd(lhs, rhs);
    }

    inline batch<double, 8> operator^(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_xor_pd(lhs, rhs);
    }

    inline batch<double, 8> operator~(const batch<double, 8>& rhs)
    {
        return _mm512_xor_pd(rhs, _mm512_castsi512_pd(_mm512_set1_epi32(-1)));
    }

    inline batch<double, 8> bitwise_andnot(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_andnot_pd(lhs, rhs);
    }

    inline batch<double, 8> min(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_min_pd(lhs, rhs);
    }

    inline batch<double, 8> max(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return _mm512_max_pd(lhs, rhs);
    }

    inline batch<double, 8> fmin(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return min(lhs, rhs);
    }

    inline batch<double, 8> fmax(const batch<double, 8>& lhs, const batch<double, 8>& rhs)
    {
        return max(lhs, rhs);
    }

    inline batch<double, 8> abs(const batch<double, 8>& rhs)
    {
        // return _mm512_abs_pd(rhs);
        return (__m512d) (_mm512_and_epi64(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF),
                                           (__m512i)((__m512d)(rhs))));
    }

    inline batch<double, 8> fabs(const batch<double, 8>& rhs)
    {
        return abs(rhs);
    }

    inline batch<double, 8> sqrt(const batch<double, 8>& rhs)
    {
        return _mm512_sqrt_pd(rhs);
    }

    inline batch<double, 8> fma(const batch<double, 8>& x, const batch<double, 8>& y, const batch<double, 8>& z)
    {
        return _mm512_fmadd_pd(x, y, z);
    }

    inline batch<double, 8> fms(const batch<double, 8>& x, const batch<double, 8>& y, const batch<double, 8>& z)
    {
        return _mm512_fmsub_pd(x, y, z);
    }

    inline batch<double, 8> fnma(const batch<double, 8>& x, const batch<double, 8>& y, const batch<double, 8>& z)
    {
        return _mm512_fnmadd_pd(x, y, z);
    }

    inline batch<double, 8> fnms(const batch<double, 8>& x, const batch<double, 8>& y, const batch<double, 8>& z)
    {
        return _mm512_fnmsub_pd(x, y, z);
    }

    inline double hadd(const batch<double, 8>& rhs)
    {
        // return _mm512_reduce_add_pd(rhs);
        __m256d tmp1 = _mm512_extractf64x4_pd(rhs, 1);
        __m256d tmp2 = _mm512_extractf64x4_pd(rhs, 0);
        __m256d res1 = tmp1 + tmp2;
        return hadd(batch<double, 4>(res1));
    }

    // inline batch<double, 8> step1(batch<double, 8> a, batch<double, 8> b)
    // {
    //     auto tmp1 = _mm512_shuffle_f64x2(a, b, _MM_SHUFFLE(1, 0, 1, 0));
    //     auto tmp2 = _mm512_shuffle_f64x2(a, b, _MM_SHUFFLE(3, 2, 3, 2));
    //     auto res = (tmp1 + tmp2);
    //     return res;
    // }

    inline batch<double, 8> haddp(const batch<double, 8>* row)
    {
    #define step1(I, a, b)                                                   \
        batch<double, 8> res ## I;                                           \
        {                                                                    \
            auto tmp1 = _mm512_shuffle_f64x2(a, b, _MM_SHUFFLE(1, 0, 1, 0)); \
            auto tmp2 = _mm512_shuffle_f64x2(a, b, _MM_SHUFFLE(3, 2, 3, 2)); \
            res ## I = (tmp1 + tmp2);                                        \
        }                                                                    \

        step1(1, row[0], row[2]);
        step1(2, row[4], row[6]);
        step1(3, row[1], row[3]);
        step1(4, row[5], row[7]);

    #undef step1

        batch<double, 8> tmp5 = _mm512_shuffle_f64x2(res1, res2, _MM_SHUFFLE(2, 0, 2, 0));
        batch<double, 8> tmp6 = _mm512_shuffle_f64x2(res1, res2, _MM_SHUFFLE(3, 1, 3, 1));

        batch<double, 8> resx1 = (tmp5 + tmp6);

        batch<double, 8> tmp7 = _mm512_shuffle_f64x2(res3, res4, _MM_SHUFFLE(2, 0, 2, 0));
        batch<double, 8> tmp8 = _mm512_shuffle_f64x2(res3, res4, _MM_SHUFFLE(3, 1, 3, 1));

        batch<double, 8> resx2 = (tmp7 + tmp8);

        batch<double, 8> tmpx = _mm512_shuffle_pd(resx1, resx2, 0b00000000);
        batch<double, 8> tmpy = _mm512_shuffle_pd(resx1, resx2, 0b11111111);

        return tmpx + tmpy;
    }

    inline batch<double, 8> select(const batch_bool<double, 8>& cond, const batch<double, 8>& a, const batch<double, 8>& b)
    {
        return _mm512_mask_blend_pd(cond, b, a);
    }

    inline batch_bool<double, 8> isnan(const batch<double, 8>& x)
    {
        return _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);
    }
}

#endif
