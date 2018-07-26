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
        static constexpr std::size_t align = 0;
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
        static constexpr std::size_t align = 64;
    };

    template <>
    class batch<double, 8> : public simd_batch<batch<double, 8>>
    {
    public:

        using self_type = batch<double, 8>;
        using base_type = simd_batch<self_type>;

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

        batch& load_aligned(const int8_t* src);
        batch& load_unaligned(const int8_t* src);

        batch& load_aligned(const uint8_t* src);
        batch& load_unaligned(const uint8_t* src);

        void store_aligned(double* dst) const;
        void store_unaligned(double* dst) const;

        void store_aligned(float* dst) const;
        void store_unaligned(float* dst) const;

        void store_aligned(int32_t* dst) const;
        void store_unaligned(int32_t* dst) const;

        void store_aligned(int64_t* dst) const;
        void store_unaligned(int64_t* dst) const;

        void store_aligned(int8_t* dst) const;
        void store_unaligned(int8_t* dst) const;

        void store_aligned(uint8_t* dst) const;
        void store_unaligned(uint8_t* dst) const;

        using base_type::load_aligned;
        using base_type::load_unaligned;
        using base_type::store_aligned;
        using base_type::store_unaligned;

        double operator[](std::size_t index) const;

    private:

        __m512d m_value;
    };

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

    inline batch<double, 8>& batch<double, 8>::load_aligned(const int8_t* src)
    {
        __m128i tmp = _mm_loadl_epi64((const __m128i*)src);
        __m512i tmp2 = _mm512_cvtepi8_epi64(tmp);
        m_value = _mm512_cvtepi64_pd(tmp2);
        return *this;
    }

    inline batch<double, 8>& batch<double, 8>::load_unaligned(const int8_t* src)
    {
        return load_aligned(src);
    }

    inline batch<double, 8>& batch<double, 8>::load_aligned(const uint8_t* src)
    {
        __m128i tmp = _mm_loadl_epi64((const __m128i*)src);
        __m512i tmp2 = _mm512_cvtepu8_epi64(tmp);
        m_value = _mm512_cvtepi64_pd(tmp2);
        return *this;
    }

    inline batch<double, 8>& batch<double, 8>::load_unaligned(const uint8_t* src)
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

    inline void batch<double, 8>::store_aligned(int8_t* dst) const
    {
        __m512i tmp = _mm512_cvtpd_epi64(m_value);
        __m128i tmp2 = _mm512_cvtepi64_epi8(tmp);
        _mm_storel_epi64((__m128i*)dst, tmp2);
    }

    inline void batch<double, 8>::store_unaligned(int8_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<double, 8>::store_aligned(uint8_t* dst) const
    {
        __m512i tmp = _mm512_cvtpd_epi64(m_value);
        __m128i tmp2 = _mm512_cvtusepi64_epi8(tmp);
        _mm_storel_epi64((__m128i*)dst, tmp2);
    }

    inline void batch<double, 8>::store_unaligned(uint8_t* dst) const
    {
        store_aligned(dst);
    }

    inline double batch<double, 8>::operator[](std::size_t index) const
    {
        alignas(64) double x[8];
        store_aligned(x);
        return x[index & 7];
    }

    namespace detail
    {
        template <>
        struct batch_kernel<double, 8>
        {
            using batch_type = batch<double, 8>;
            using value_type = double;
            using batch_bool_type = batch_bool<double, 8>;

            static batch_type neg(const batch_type& rhs)
            {
                return _mm512_xor_pd(rhs, _mm512_castsi512_pd(_mm512_set1_epi64(0x8000000000000000)));
            }

            static batch_type add(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_add_pd(lhs, rhs);
            }

            static batch_type sub(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_sub_pd(lhs, rhs);
            }

            static batch_type mul(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_mul_pd(lhs, rhs);
            }

            static batch_type div(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_div_pd(lhs, rhs);
            }

            static batch_bool_type eq(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmp_pd_mask(lhs, rhs, _CMP_EQ_OQ);
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmp_pd_mask(lhs, rhs, _CMP_NEQ_OQ);
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmp_pd_mask(lhs, rhs, _CMP_LT_OQ);
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_cmp_pd_mask(lhs, rhs, _CMP_LE_OQ);
            }

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_and_pd(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_or_pd(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_xor_pd(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return _mm512_xor_pd(rhs, _mm512_castsi512_pd(_mm512_set1_epi32(-1)));
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_andnot_pd(lhs, rhs);
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_min_pd(lhs, rhs);
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_max_pd(lhs, rhs);
            }

            static batch_type fmin(const batch_type& lhs, const batch_type& rhs)
            {
                return min(lhs, rhs);
            }

            static batch_type fmax(const batch_type& lhs, const batch_type& rhs)
            {
                return max(lhs, rhs);
            }

            static batch_type abs(const batch_type& rhs)
            {
                return (__m512d)(_mm512_and_epi64(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF),
                                                  (__m512i)((__m512d)(rhs))));
            }

            static batch_type fabs(const batch_type& rhs)
            {
                return abs(rhs);
            }

            static batch_type sqrt(const batch_type& rhs)
            {
                return _mm512_sqrt_pd(rhs);
            }

            static batch_type fma(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return _mm512_fmadd_pd(x, y, z);
            }

            static batch_type fms(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return _mm512_fmsub_pd(x, y, z);
            }

            static batch_type fnma(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return _mm512_fnmadd_pd(x, y, z);
            }

            static batch_type fnms(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return _mm512_fnmsub_pd(x, y, z);
            }

            static value_type hadd(const batch_type& rhs)
            {
                __m256d tmp1 = _mm512_extractf64x4_pd(rhs, 1);
                __m256d tmp2 = _mm512_extractf64x4_pd(rhs, 0);
                __m256d res1 = tmp1 + tmp2;
                return xsimd::hadd(batch<double, 4>(res1));
            }

            static batch_type haddp(const simd_batch<batch_type>* row)
            {
#define step1(I, a, b)                                                   \
        batch<double, 8> res ## I;                                           \
        {                                                                    \
            auto tmp1 = _mm512_shuffle_f64x2(a, b, _MM_SHUFFLE(1, 0, 1, 0)); \
            auto tmp2 = _mm512_shuffle_f64x2(a, b, _MM_SHUFFLE(3, 2, 3, 2)); \
            res ## I = (tmp1 + tmp2);                                        \
        }                                                                    \

                step1(1, row[0](), row[2]());
                step1(2, row[4](), row[6]());
                step1(3, row[1](), row[3]());
                step1(4, row[5](), row[7]());

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

            static batch_type select(const batch_bool_type& cond, const batch_type& a, const batch_type& b)
            {
                return _mm512_mask_blend_pd(cond, b, a);
            }

            static batch_bool_type isnan(const batch_type& x)
            {
                return _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);
            }
        };
    }
}

#endif
