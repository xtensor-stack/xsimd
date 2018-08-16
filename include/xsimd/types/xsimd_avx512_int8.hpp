/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_INT8_HPP
#define XSIMD_AVX512_INT8_HPP

#include <cstdint>

#include "xsimd_base.hpp"

namespace xsimd
{

    /****************************
     * batch_bool<int8 / int16> *
     ****************************/

    template <>
    struct simd_batch_traits<batch_bool<int8_t, 64>>
    {
        using value_type = int8_t;
        static constexpr std::size_t size = 64;
        using batch_type = batch<int8_t, 64>;
        static constexpr std::size_t align = 64;
    };

    template <>
    struct simd_batch_traits<batch_bool<uint8_t, 64>>
    {
        using value_type = uint8_t;
        static constexpr std::size_t size = 64;
        using batch_type = batch<uint8_t, 64>;
        static constexpr std::size_t align = 64;
    };

    template <>
    struct simd_batch_traits<batch_bool<int16_t, 32>>
    {
        using value_type = int16_t;
        static constexpr std::size_t size = 32;
        using batch_type = batch<int16_t, 32>;
        static constexpr std::size_t align = 64;
    };

    template <>
    struct simd_batch_traits<batch_bool<uint16_t, 32>>
    {
        using value_type = uint16_t;
        static constexpr std::size_t size = 32;
        using batch_type = batch<uint16_t, 32>;
        static constexpr std::size_t align = 64;
    };

#define XSIMD_SPLIT_AVX512(avx_name)                                                              \
    __m256i avx_name##_low = _mm512_castsi512_si256(avx_name);                                    \
    __m256i avx_name##_high = _mm512_extracti64x4_epi64(avx_name, 1)                              \

#define XSIMD_RETURN_MERGED_AVX(res_low, res_high)                                                \
    __m512i result = _mm512_castsi256_si512(res_low);                                             \
    return _mm512_inserti64x4(result, res_high, 1)                                                \

#define XSIMD_APPLY_AVX2_FUNCTION(func, avx_lhs, avx_rhs)                                             \
    XSIMD_SPLIT_AVX512(avx_lhs);                                                                      \
    XSIMD_SPLIT_AVX512(avx_rhs);                                                                      \
    __m256i res_low = detail::batch_kernel<value_type, 32> :: func (avx_lhs##_low, avx_rhs##_low);    \
    __m256i res_high = detail::batch_kernel<value_type, 32> :: func (avx_lhs##_high, avx_rhs##_high); \
    XSIMD_RETURN_MERGED_AVX(res_low, res_high);

    namespace detail_avx512
    {
        // From stackoverflow: https://stackoverflow.com/a/15908420/1347553
        template<unsigned... Is>
        struct rseq
        {
            using type = rseq;
        };

        template<unsigned I, unsigned... Is>
        struct rgen_seq
            : rgen_seq<I - 1, Is..., I - 1>
        {
        };

        template<unsigned... Is>
        struct rgen_seq<0, Is...>
            : rseq<Is...>
        {
        };

        template<class Tup, unsigned... Is>
        __m512i revert_args_set_epi8(Tup&& t, rseq<Is...>)
        {
            // funny, this instruction is not yet implemented in clang or gcc (will come in future versions)
#if defined(__clang__) || __GNUC__
            return __extension__ (__m512i)(__v64qi)
            {
                std::get<Is>(std::forward<Tup>(t))...
            };
#else
            return _mm512_set_epi8(std::get<Is>(std::forward<Tup>(t))...);
#endif
        }

        template<class Tup, unsigned... Is>
        __m512i revert_args_set_epi16(Tup&& t, rseq<Is...>)
        {
#if defined(__clang__) || __GNUC__
            return __extension__ (__m512i)(__v32hi)
            {
                std::get<Is>(std::forward<Tup>(t))...
            };
#else
            return _mm512_set_epi16(std::get<Is>(std::forward<Tup>(t))...);
#endif
        }

        template <class... Args>
        __m512i int_init(std::integral_constant<std::size_t, 1>, Args... args)
        {
            return revert_args_set_epi8(std::forward_as_tuple(args...), rgen_seq<sizeof...(Args)>{});
        }

        template <class... Args>
        __m512i int_init(std::integral_constant<std::size_t, 2>, Args... args)
        {
            return revert_args_set_epi16(std::forward_as_tuple(args...), rgen_seq<sizeof...(Args)>{});
        }
    }

#if defined(XSIMD_AVX512BW_AVAILABLE)
    template <>
    class batch_bool<int8_t, 64> :
        public batch_bool_avx512<__mmask64, batch_bool<int8_t, 64>>,
        public simd_batch_bool<batch_bool<int8_t, 64>>
    {
    public:

        using base_class = batch_bool_avx512<__mmask64, batch_bool<int8_t, 64>>;
        using base_class::base_class;
    };

    template <>
    class batch_bool<uint8_t, 64> :
        public batch_bool_avx512<__mmask64, batch_bool<uint8_t, 64>>,
        public simd_batch_bool<batch_bool<uint8_t, 64>>
    {
    public:

        using base_class = batch_bool_avx512<__mmask64, batch_bool<uint8_t, 64>>;
        using base_class::base_class;
    };

    namespace detail
    {
        template <>
        struct batch_bool_kernel<int8_t, 64>
            : batch_bool_kernel_avx512<int8_t, 64>
        {
        };

        template <>
        struct batch_bool_kernel<uint8_t, 64>
            : batch_bool_kernel_avx512<uint8_t, 64>
        {
        };
    }

#else

    template <class T, std::size_t N>
    class avx512_fallback_batch_bool : 
        public simd_batch_bool<batch_bool<T, N>>
    {
    public:

        avx512_fallback_batch_bool();
        explicit avx512_fallback_batch_bool(bool b);
        template <class... Args, class Enable = detail::is_array_initializer_t<bool, N, Args...>>
        avx512_fallback_batch_bool(Args... args);

        avx512_fallback_batch_bool(const __m512i& rhs);
        avx512_fallback_batch_bool& operator=(const __m512i& rhs);

        operator __m512i() const;

        bool operator[](std::size_t index) const;

    private:

        __m512i m_value;
    };

    template <class T, std::size_t N>
    inline avx512_fallback_batch_bool<T, N>::avx512_fallback_batch_bool()
    {
    }

    template <class T, std::size_t N>
    inline avx512_fallback_batch_bool<T, N>::avx512_fallback_batch_bool(bool b)
        : m_value(_mm512_set1_epi64(-(int64_t)b))
    {
    }

    template <class T, std::size_t N>
    template <class... Args, class>
    inline avx512_fallback_batch_bool<T, N>::avx512_fallback_batch_bool(Args... args)
        : m_value(detail_avx512::int_init(std::integral_constant<std::size_t, sizeof(int8_t)>{},
                  static_cast<int8_t>(-static_cast<bool>(args))...))
    {
    }

    template <class T, std::size_t N>
    inline avx512_fallback_batch_bool<T, N>::avx512_fallback_batch_bool(const __m512i& rhs)
        : m_value(rhs)
    {
    }

    template <class T, std::size_t N>
    inline avx512_fallback_batch_bool<T, N>::operator __m512i() const
    {
        return m_value;
    }

    template <class T, std::size_t N>
    inline avx512_fallback_batch_bool<T, N>& avx512_fallback_batch_bool<T, N>::operator=(const __m512i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    template <class T, std::size_t N>
    inline bool avx512_fallback_batch_bool<T, N>::operator[](std::size_t idx) const
    {
        alignas(64) T x[N];
        _mm512_store_si512((__m512i*) x, m_value);
        return x[idx & (N - 1)];
    }

    namespace detail
    {
        template <class T, std::size_t N>
        struct avx512_fallback_batch_bool_kernel
        {
            using batch_type = batch_bool<T, N>;

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_and_si512(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_or_si512(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_xor_si512(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return _mm512_xor_si512(rhs, _mm512_set1_epi64(-1)); // xor with all one
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_andnot_si512(lhs, rhs);
            }

            static batch_type equal(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(lhs ^ rhs);
            }

            static batch_type not_equal(const batch_type& lhs, const batch_type& rhs)
            {
                return lhs ^ rhs;
            }

            static bool all(const batch_type& rhs)
            {
                XSIMD_SPLIT_AVX512(rhs);
                bool res_hi = _mm256_testc_si256(rhs_high, batch_bool<int32_t, 8>(true)) != 0;
                bool res_lo = _mm256_testc_si256(rhs_low, batch_bool<int32_t, 8>(true)) != 0;
                return res_hi && res_lo;
            }

            static bool any(const batch_type& rhs)
            {
                XSIMD_SPLIT_AVX512(rhs);
                bool res_hi = !_mm256_testz_si256(rhs_high, rhs_high);
                bool res_lo = !_mm256_testz_si256(rhs_low, rhs_low);
                return res_hi || res_lo;
            }
        };

        template <>
        struct batch_bool_kernel<int8_t, 64>
            : avx512_fallback_batch_bool_kernel<int8_t, 64>
        {
        };

        template <>
        struct batch_bool_kernel<uint8_t, 64>
            : avx512_fallback_batch_bool_kernel<int8_t, 64>
        {
        };
    }
#endif

    /*********************
     * batch<int32_t, 8> *
     *********************/

    template <>
    struct simd_batch_traits<batch<int8_t, 64>>
    {
        using value_type = int8_t;
        static constexpr std::size_t size = 64;
        using batch_bool_type = batch_bool<int8_t, 64>;
        static constexpr std::size_t align = 64;
    };

    template <>
    struct simd_batch_traits<batch<uint8_t, 64>>
    {
        using value_type = uint8_t;
        static constexpr std::size_t size = 64;
        using batch_bool_type = batch_bool<uint8_t, 64>;
        static constexpr std::size_t align = 64;
    };

    template <>
    struct simd_batch_traits<batch<int16_t, 32>>
    {
        using value_type = int16_t;
        static constexpr std::size_t size = 32;
        using batch_bool_type = batch_bool<int16_t, 32>;
        static constexpr std::size_t align = 64;
    };

    template <>
    struct simd_batch_traits<batch<uint16_t, 32>>
    {
        using value_type = uint16_t;
        static constexpr std::size_t size = 32;
        using batch_bool_type = batch_bool<uint16_t, 32>;
        static constexpr std::size_t align = 64;
    };

    template <class T, std::size_t N>
    class avx512_int_batch : public simd_batch<batch<T, N>>
    {
    public:

        avx512_int_batch();
        explicit avx512_int_batch(T i);

        template <class... Args, class Enable = detail::is_array_initializer_t<T, N, Args...>>
        avx512_int_batch(Args... exactly_N_scalars);
        explicit avx512_int_batch(const T* src);
        avx512_int_batch(const T* src, aligned_mode);
        avx512_int_batch(const T* src, unaligned_mode);

        avx512_int_batch(const __m512i& rhs);
        avx512_int_batch& operator=(const __m512i& rhs);

        operator __m512i() const;

        batch<T, N>& load_aligned(const T* src);
        batch<T, N>& load_unaligned(const T* src);

        void store_aligned(T* dst) const;
        void store_unaligned(T* dst) const;

        int operator[](std::size_t index) const;

    private:

        __m512i m_value;
    };

    template <>
    class batch<int8_t, 64> : public avx512_int_batch<int8_t, 64>
    {
    public:

        using base_class = avx512_int_batch;
        using base_class::base_class;
        using base_class::load_aligned;
        using base_class::load_unaligned;
        using base_class::store_aligned;
        using base_class::store_unaligned;

        batch() = default;

        explicit batch(const char* src)
            : batch(reinterpret_cast<const int8_t*>(src))
        {
        }

        batch(const char* src, aligned_mode)
            : batch(reinterpret_cast<const int8_t*>(src), aligned_mode{})
        {
        }

        batch(const char* src, unaligned_mode)
            : batch(reinterpret_cast<const int8_t*>(src), unaligned_mode{})
        {
        }

        batch& load_aligned(const char* src)
        {
            return load_aligned(reinterpret_cast<const int8_t*>(src));
        }

        batch& load_unaligned(const char* src)
        {
            return load_unaligned(reinterpret_cast<const int8_t*>(src));
        }

        void store_aligned(char* dst)
        {
            return store_aligned(reinterpret_cast<int8_t*>(dst));
        }

        void store_unaligned(char* dst)
        {
            return store_unaligned(reinterpret_cast<int8_t*>(dst));
        }
    };

    template <>
    class batch<uint8_t, 64> : public avx512_int_batch<uint8_t, 64>
    {
    public:

        using avx512_int_batch::avx512_int_batch;
    };

    // template <>
    // class batch<int16_t, 16> : public avx512_int_batch<int16_t, 16>
    // {
    // public:

    //     using avx512_int_batch::avx512_int_batch;
    // };

    // template <>
    // class batch<uint16_t, 16> : public avx512_int_batch<uint16_t, 16>
    // {
    // public:

    //     using avx512_int_batch::avx512_int_batch;
    // };

    // batch<int32_t, 8> operator<<(const batch<int32_t, 8>& lhs, int32_t rhs);
    // batch<int32_t, 8> operator>>(const batch<int32_t, 8>& lhs, int32_t rhs);

    /**************************************
     * avx512_int_batch<T, N> implementation *
     **************************************/


    template <class T, std::size_t N>
    inline avx512_int_batch<T, N>::avx512_int_batch()
    {
    }

    template <class T, std::size_t N>
    inline avx512_int_batch<T, N>::avx512_int_batch(T i)
        : m_value(sizeof(T) == 1 ? _mm512_set1_epi8(i)  :
                  sizeof(T) == 2 ? _mm512_set1_epi16(i) :
                  sizeof(T) == 4 ? _mm512_set1_epi32(i) : _mm512_set1_epi64(i))
    {
    }

    template <class T, std::size_t N>
    template <class... Args, class>
    inline avx512_int_batch<T, N>::avx512_int_batch(Args... args)
        : m_value(detail_avx512::int_init(std::integral_constant<std::size_t, sizeof(T)>{}, args...))
    {
    }

    template <class T, std::size_t N>
    inline avx512_int_batch<T, N>::avx512_int_batch(const T* src)
        : m_value(_mm512_loadu_si512((__m512i const*) src))
    {
    }

    template <class T, std::size_t N>
    inline avx512_int_batch<T, N>::avx512_int_batch(const T* src, aligned_mode)
        : m_value(_mm512_load_si512((__m512i const*) src))
    {
    }

    template <class T, std::size_t N>
    inline avx512_int_batch<T, N>::avx512_int_batch(const T* src, unaligned_mode)
        : m_value(_mm512_loadu_si512((__m512i const*) src))
    {
    }

    template <class T, std::size_t N>
    inline avx512_int_batch<T, N>::avx512_int_batch(const __m512i& rhs)
        : m_value(rhs)
    {
    }

    template <class T, std::size_t N>
    inline avx512_int_batch<T, N>& avx512_int_batch<T, N>::operator=(const __m512i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    template <class T, std::size_t N>
    inline avx512_int_batch<T, N>::operator __m512i() const
    {
        return m_value;
    }

    template <class T, std::size_t N>
    inline batch<T, N>& avx512_int_batch<T, N>::load_aligned(const T* src)
    {
        m_value = _mm512_load_si512((__m512i const*) src);
        return (*this)();
    }

    template <class T, std::size_t N>
    inline batch<T, N>& avx512_int_batch<T, N>::load_unaligned(const T* src)
    {
        m_value = _mm512_loadu_si512((__m512i const*) src);
        return (*this)();
    }

    template <class T, std::size_t N>
    inline void avx512_int_batch<T, N>::store_aligned(T* dst) const
    {
        _mm512_store_si512((__m512i*) dst, m_value);
    }

    template <class T, std::size_t N>
    inline void avx512_int_batch<T, N>::store_unaligned(T* dst) const
    {
        _mm512_storeu_si512((__m512i*) dst, m_value);
    }

    template <class T, std::size_t N>
    inline int avx512_int_batch<T, N>::operator[](std::size_t index) const
    {
        alignas(64) T x[N];
        store_aligned(x);
        return x[index & (N - 1)];
    }

    namespace detail
    {
        template <class T>
        struct avx512_int8_batch_kernel
        {
            using batch_type = batch<T, 64>;
            using value_type = T;
            using batch_bool_type = batch_bool<T, 64>;

            static batch_type neg(const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_sub_epi8(_mm512_setzero_si512(), rhs);
            #else
                XSIMD_SPLIT_AVX512(rhs);
                __m256i res_low = _mm256_sub_epi8(_mm256_setzero_si256(), rhs_low);
                __m256i res_high = _mm256_sub_epi8(_mm256_setzero_si256(), rhs_high);
                XSIMD_RETURN_MERGED_AVX(res_low, res_high);
            #endif
            }

            static batch_type add(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_add_epi8(lhs, rhs);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(add, lhs, rhs);
            #endif
            }

            static batch_type sub(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_sub_epi8(lhs, rhs);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(sub, lhs, rhs);
            #endif
            }

            static batch_type mul(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                batch_type upper = _mm512_and_si512(_mm512_mullo_epi16(lhs, rhs), _mm512_srli_epi16(_mm512_set1_epi16(-1), 8));
                batch_type lower = _mm512_slli_epi16(_mm512_mullo_epi16(_mm512_srli_epi16(lhs, 8), _mm512_srli_epi16(rhs, 8)), 8);
                return _mm512_or_si512(upper, lower);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(mul, lhs, rhs);
            #endif
            }

            static batch_type div(const batch_type& lhs, const batch_type& rhs)
            {
                XSIMD_APPLY_AVX2_FUNCTION(div, lhs, rhs);
            }

            static batch_type mod(const batch_type& lhs, const batch_type& rhs)
            {
                XSIMD_MACRO_UNROLL_BINARY(%);
            }

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_and_si512(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_or_si512(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_xor_si512(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return _mm512_xor_si512(rhs, _mm512_set1_epi8(-1));
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm512_andnot_si512(lhs, rhs);
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
                XSIMD_SPLIT_AVX512(rhs);
                auto tmp = batch<value_type, 32>(rhs_low) + batch<value_type, 32>(rhs_high);
                return xsimd::hadd(batch<value_type, 32>(tmp));
            }

            static batch_type select(const batch_bool_type& cond, const batch_type& a, const batch_type& b)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_mask_blend_epi8(cond, b, a);
            #else
                XSIMD_SPLIT_AVX512(cond);
                XSIMD_SPLIT_AVX512(a);
                XSIMD_SPLIT_AVX512(b);

                auto res_lo = _mm256_blendv_epi8(b_low, a_low, cond_low);
                auto res_hi = _mm256_blendv_epi8(b_high, a_high, cond_high);

                XSIMD_RETURN_MERGED_AVX(res_lo, res_hi);
            #endif
            }
        };

        template <>
        struct batch_kernel<int8_t, 64>
            : public avx512_int8_batch_kernel<int8_t>
        {
            static batch_type abs(const batch_type& rhs)
            {
                return _mm512_abs_epi8(rhs);
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_min_epi8(lhs, rhs);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(min, lhs, rhs);
            #endif
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_max_epi8(lhs, rhs);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(max, lhs, rhs);
            #endif
            }

            static batch_bool_type eq(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_cmp_epi8_mask(lhs, rhs, _MM_CMPINT_EQ);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(eq, lhs, rhs);
            #endif
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_cmp_epi8_mask(lhs, rhs, _MM_CMPINT_NE);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(neq, lhs, rhs);
            #endif
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_cmp_epi8_mask(lhs, rhs, _MM_CMPINT_LT);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(lt, lhs, rhs);
            #endif
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_cmp_epi8_mask(lhs, rhs, _MM_CMPINT_LE);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(lte, lhs, rhs);
            #endif
            }
        };

        template <>
        struct batch_kernel<uint8_t, 64>
            : public avx512_int8_batch_kernel<uint8_t>
        {
            static batch_type abs(const batch_type& rhs)
            {
                return rhs;
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_min_epu8(lhs, rhs);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(min, lhs, rhs);
            #endif
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_max_epu8(lhs, rhs);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(max, lhs, rhs);
            #endif
            }

            static batch_bool_type eq(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_cmp_epu8_mask(lhs, rhs, _MM_CMPINT_EQ);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(eq, lhs, rhs);
            #endif
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_cmp_epu8_mask(lhs, rhs, _MM_CMPINT_NE);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(neq, lhs, rhs);
            #endif
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_cmp_epu8_mask(lhs, rhs, _MM_CMPINT_LT);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(lt, lhs, rhs);
            #endif
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
            #if defined(XSIMD_AVX512BW_AVAILABLE)
                return _mm512_cmp_epu8_mask(lhs, rhs, _MM_CMPINT_LE);
            #else
                XSIMD_APPLY_AVX2_FUNCTION(lte, lhs, rhs);
            #endif
            }
        };
    }

    namespace avx512_detail
    {
        template <class F, class T, std::size_t N>
        inline batch<T, N> shift_impl(F&& f, const batch<T, N>& lhs, int32_t rhs)
        {
            alignas(64) T tmp_lhs[N], tmp_res[N];
            lhs.store_aligned(&tmp_lhs[0]);
            unroller<N>([&](std::size_t i) {
                tmp_res[i] = f(tmp_lhs[i], rhs);
            });
            return batch<T, N>(tmp_res, aligned_mode());
        }
    }

    // Not yet implemented, need to convert to int16
    inline batch<int8_t, 64> operator<<(const batch<int8_t, 64>& lhs, int32_t rhs)
    {
        return avx_detail::shift_impl([](int8_t val, int32_t rhs) {
            return val << rhs;
        }, lhs, rhs);
    }

    inline batch<int8_t, 64> operator>>(const batch<int8_t, 64>& lhs, int32_t rhs)
    {
        return avx_detail::shift_impl([](int8_t val, int32_t rhs) {
            return val >> rhs;
        }, lhs, rhs);
    }

    inline batch<uint8_t, 64> operator<<(const batch<uint8_t, 64>& lhs, int32_t rhs)
    {
        return avx_detail::shift_impl([](uint8_t val, int32_t rhs) {
            return val << rhs;
        }, lhs, rhs);
    }

    inline batch<uint8_t, 64> operator>>(const batch<uint8_t, 64>& lhs, int32_t rhs)
    {
        return avx_detail::shift_impl([](uint8_t val, int32_t rhs) {
            return val >> rhs;
        }, lhs, rhs);
    }
}

#undef XSIMD_SPLIT_AVX512
#undef XSIMD_RETURN_MERGED_AVX
#undef XSIMD_APPLY_AVX2_FUNCTION

#endif
