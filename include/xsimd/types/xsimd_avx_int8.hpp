/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX_INT8_HPP
#define XSIMD_AVX_INT8_HPP

#include <cstdint>

#include "xsimd_base.hpp"

namespace xsimd
{

    /****************************
     * batch_bool<int8 / int16> *
     ****************************/

    template <>
    struct simd_batch_traits<batch_bool<int8_t, 32>>
    {
        using value_type = int8_t;
        static constexpr std::size_t size = 32;
        using batch_type = batch<int8_t, 32>;
        static constexpr std::size_t align = 32;
    };

    template <>
    struct simd_batch_traits<batch_bool<uint8_t, 32>>
    {
        using value_type = uint8_t;
        static constexpr std::size_t size = 32;
        using batch_type = batch<uint8_t, 32>;
        static constexpr std::size_t align = 32;
    };

    template <>
    struct simd_batch_traits<batch_bool<int16_t, 16>>
    {
        using value_type = int16_t;
        static constexpr std::size_t size = 16;
        using batch_type = batch<int16_t, 16>;
        static constexpr std::size_t align = 32;
    };

    template <>
    struct simd_batch_traits<batch_bool<uint16_t, 16>>
    {
        using value_type = uint16_t;
        static constexpr std::size_t size = 16;
        using batch_type = batch<uint16_t, 16>;
        static constexpr std::size_t align = 32;
    };

    template <class T, std::size_t N>
    class avx_int_batch_bool : public simd_batch_bool<batch_bool<T, N>>
    {
    public:

        avx_int_batch_bool();
        explicit avx_int_batch_bool(bool b);
        template <class... Args, class Enable = detail::is_array_initializer_t<bool, N, Args...>>
        avx_int_batch_bool(Args... args);

        avx_int_batch_bool(const __m256i& rhs);
        avx_int_batch_bool& operator=(const __m256i& rhs);

        operator __m256i() const;

        bool operator[](std::size_t index) const;

    private:

        __m256i m_value;
    };

    template <>
    class batch_bool<int8_t, 32> : public avx_int_batch_bool<int8_t, 32>
    {
    public:
        using avx_int_batch_bool::avx_int_batch_bool;
    };

    template <>
    class batch_bool<uint8_t, 32> : public avx_int_batch_bool<uint8_t, 32>
    {
    public:
        using avx_int_batch_bool::avx_int_batch_bool;
    };

    /*********************
     * batch<int32_t, 8> *
     *********************/

    template <>
    struct simd_batch_traits<batch<int8_t, 32>>
    {
        using value_type = int8_t;
        static constexpr std::size_t size = 32;
        using batch_bool_type = batch_bool<int8_t, 32>;
        static constexpr std::size_t align = 32;
    };

    template <>
    struct simd_batch_traits<batch<uint8_t, 32>>
    {
        using value_type = uint8_t;
        static constexpr std::size_t size = 32;
        using batch_bool_type = batch_bool<uint8_t, 32>;
        static constexpr std::size_t align = 32;
    };

    template <>
    struct simd_batch_traits<batch<int16_t, 16>>
    {
        using value_type = int16_t;
        static constexpr std::size_t size = 16;
        using batch_bool_type = batch_bool<int16_t, 16>;
        static constexpr std::size_t align = 32;
    };

    template <>
    struct simd_batch_traits<batch<uint16_t, 16>>
    {
        using value_type = uint16_t;
        static constexpr std::size_t size = 16;
        using batch_bool_type = batch_bool<uint16_t, 16>;
        static constexpr std::size_t align = 32;
    };

    template <class T, std::size_t N>
    class avx_int_batch : public simd_batch<batch<T, N>>
    {
    public:

        avx_int_batch();
        explicit avx_int_batch(T i);
        // Constructor from N scalar parameters
        template <class... Args, class Enable = detail::is_array_initializer_t<T, N, Args...>>
        avx_int_batch(Args... exactly_N_scalars);

        explicit avx_int_batch(const T* src);
        avx_int_batch(const T* src, aligned_mode);
        avx_int_batch(const T* src, unaligned_mode);
        avx_int_batch(const __m256i& rhs);
        avx_int_batch& operator=(const __m256i& rhs);

        operator __m256i() const;

        batch<T, N>& load_aligned(const T* src);
        batch<T, N>& load_unaligned(const T* src);

        void store_aligned(T* dst) const;
        void store_unaligned(T* dst) const;

        int operator[](std::size_t index) const;

    private:

        __m256i m_value;
    };

    template <>
    class batch<int8_t, 32> : public avx_int_batch<int8_t, 32>
    {
    public:

        using base_class = avx_int_batch;
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
    class batch<uint8_t, 32> : public avx_int_batch<uint8_t, 32>
    {
    public:
        using avx_int_batch::avx_int_batch;
    };

    template <>
    class batch<int16_t, 16> : public avx_int_batch<int16_t, 16>
    {
    public:
        using avx_int_batch::avx_int_batch;
    };

    template <>
    class batch<uint16_t, 16> : public avx_int_batch<uint16_t, 16>
    {
    public:
        using avx_int_batch::avx_int_batch;
    };

    batch<int8_t, 32> operator<<(const batch<int8_t, 32>& lhs, int32_t rhs);
    batch<int8_t, 32> operator>>(const batch<int8_t, 32>& lhs, int32_t rhs);
    batch<uint8_t, 32> operator<<(const batch<uint8_t, 32>& lhs, int32_t rhs);
    batch<uint8_t, 32> operator>>(const batch<uint8_t, 32>& lhs, int32_t rhs);

    namespace avx_detail
    {
        template <class... Args>
        inline __m256i int_init(std::integral_constant<std::size_t, 1>, Args... args)
        {
            return _mm256_setr_epi8(args...);
        }

        template <class... Args>
        inline __m256i int_init(std::integral_constant<std::size_t, 2>, Args... args)
        {
            return _mm256_setr_epi16(args...);
        }
    }

    /*****************************************
     * batch_bool<T, N> implementation *
     *****************************************/

#if XSIMD_X86_INSTR_SET < XSIMD_X86_AVX2_VERSION

#define XSIMD_SPLIT_AVX(avx_name)                              \
    __m128i avx_name##_low = _mm256_castsi256_si128(avx_name); \
    __m128i avx_name##_high = _mm256_extractf128_si256(avx_name, 1)

#define XSIMD_RETURN_MERGED_SSE(res_low, res_high)    \
    __m256i result = _mm256_castsi128_si256(res_low); \
    return _mm256_insertf128_si256(result, res_high, 1)

#define XSIMD_APPLY_SSE_FUNCTION(func, avx_lhs, avx_rhs)     \
    XSIMD_SPLIT_AVX(avx_lhs);                                \
    XSIMD_SPLIT_AVX(avx_rhs);                                \
    __m128i res_low = func(avx_lhs##_low, avx_rhs##_low);    \
    __m128i res_high = func(avx_lhs##_high, avx_rhs##_high); \
    XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif

    template <class T, std::size_t N>
    inline avx_int_batch_bool<T, N>::avx_int_batch_bool()
    {
    }

    template <class T, std::size_t N>
    inline avx_int_batch_bool<T, N>::avx_int_batch_bool(bool b)
        : m_value(_mm256_set1_epi32(-(int32_t)b))
    {
    }

    template <class T, std::size_t N>
    template <class... Args, class>
    inline avx_int_batch_bool<T, N>::avx_int_batch_bool(Args... args)
        : m_value(avx_detail::int_init(std::integral_constant<std::size_t, sizeof(T)>{}, -static_cast<T>(static_cast<bool>(args))...))
    {
    }

    template <class T, std::size_t N>
    inline avx_int_batch_bool<T, N>::avx_int_batch_bool(const __m256i& rhs)
        : m_value(rhs)
    {
    }

    template <class T, std::size_t N>
    inline avx_int_batch_bool<T, N>& avx_int_batch_bool<T, N>::operator=(const __m256i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    template <class T, std::size_t N>
    inline avx_int_batch_bool<T, N>::operator __m256i() const
    {
        return m_value;
    }

    template <class T, std::size_t N>
    inline bool avx_int_batch_bool<T, N>::operator[](std::size_t index) const
    {
        alignas(32) T x[N];
        _mm256_store_si256((__m256i*)x, m_value);
        return static_cast<bool>(x[index & (N - 1)]);
    }

    namespace detail
    {
        template <class T, std::size_t N>
        struct avx_int_batch_bool_kernel
        {
            using batch_type = batch_bool<T, N>;

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_and_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_and_si128, lhs, rhs);
#endif
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_or_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_or_si128, lhs, rhs);
#endif
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_xor_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_xor_si128, lhs, rhs);
#endif
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_xor_si256(rhs, _mm256_set1_epi32(-1)); // xor with all one
#else
                XSIMD_SPLIT_AVX(rhs);
                __m128i res_low = _mm_xor_si128(rhs_low, _mm_set1_epi32(-1));
                __m128i res_high = _mm_xor_si128(rhs_high, _mm_set1_epi32(-1));
                XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_andnot_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_andnot_si128, lhs, rhs);
#endif
            }

            static batch_type equal(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                switch(sizeof(T))
                {
                    case 1:
                        return _mm256_cmpeq_epi8(lhs, rhs);
                    case 2:
                        return _mm256_cmpeq_epi16(lhs, rhs);
                    case 4:
                        return _mm256_cmpeq_epi32(lhs, rhs);
                    case 8:
                        return _mm256_cmpeq_epi64(lhs, rhs);
                }
#else
                switch(sizeof(T))
                {
                    case 1:
                    {
                        XSIMD_APPLY_SSE_FUNCTION(_mm_cmpeq_epi8, lhs, rhs);
                    }
                    case 2:
                    {
                        XSIMD_APPLY_SSE_FUNCTION(_mm_cmpeq_epi16, lhs, rhs);
                    }
                    case 4:
                    {
                        XSIMD_APPLY_SSE_FUNCTION(_mm_cmpeq_epi32, lhs, rhs);
                    }
                    case 8:
                    {
                        XSIMD_APPLY_SSE_FUNCTION(_mm_cmpeq_epi64, lhs, rhs);
                    }
                }
#endif
            }

            static batch_type not_equal(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(lhs == rhs);
            }

            static bool all(const batch_type& rhs)
            {
                return _mm256_testc_si256(rhs, batch_bool<int32_t, 8>(true)) != 0;
            }

            static bool any(const batch_type& rhs)
            {
                return !_mm256_testz_si256(rhs, rhs);
            }
        };

        template <>
        struct batch_bool_kernel<int8_t, 32> : public avx_int_batch_bool_kernel<int8_t, 32>
        {
        };

        template <>
        struct batch_bool_kernel<uint8_t, 32> : public avx_int_batch_bool_kernel<uint8_t, 32>
        {
        };
    }

    /**************************************
     * avx_int_batch<T, N> implementation *
     **************************************/

    template <class T, std::size_t N>
    inline avx_int_batch<T, N>::avx_int_batch()
    {
    }

    template <class T, std::size_t N>
    inline avx_int_batch<T, N>::avx_int_batch(T i)
        : m_value(sizeof(T) == 1 ? _mm256_set1_epi8(i)  :
                  sizeof(T) == 2 ? _mm256_set1_epi16(i) :
                  sizeof(T) == 4 ? _mm256_set1_epi32(i) : _mm256_set1_epi64x(i))
    {
    }

    template <class T, std::size_t N>
    template <class... Args, class>
    inline avx_int_batch<T, N>::avx_int_batch(Args... args)
        : m_value(avx_detail::int_init(std::integral_constant<std::size_t, sizeof(T)>{}, args...))
    {
    }

    template <class T, std::size_t N>
    inline avx_int_batch<T, N>::avx_int_batch(const T* src)
        : m_value(_mm256_loadu_si256((__m256i const*)src))
    {
    }

    template <class T, std::size_t N>
    inline avx_int_batch<T, N>::avx_int_batch(const T* src, aligned_mode)
        : m_value(_mm256_load_si256((__m256i const*)src))
    {
    }

    template <class T, std::size_t N>
    inline avx_int_batch<T, N>::avx_int_batch(const T* src, unaligned_mode)
        : m_value(_mm256_loadu_si256((__m256i const*)src))
    {
    }

    template <class T, std::size_t N>
    inline avx_int_batch<T, N>::avx_int_batch(const __m256i& rhs)
        : m_value(rhs)
    {
    }

    template <class T, std::size_t N>
    inline avx_int_batch<T, N>& avx_int_batch<T, N>::operator=(const __m256i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    template <class T, std::size_t N>
    inline avx_int_batch<T, N>::operator __m256i() const
    {
        return m_value;
    }

    template <class T, std::size_t N>
    inline batch<T, N>& avx_int_batch<T, N>::load_aligned(const T* src)
    {
        m_value = _mm256_load_si256((__m256i const*) src);
        return (*this)();
    }

    template <class T, std::size_t N>
    inline batch<T, N>& avx_int_batch<T, N>::load_unaligned(const T* src)
    {
        m_value = _mm256_loadu_si256((__m256i const*) src);
        return (*this)();
    }

    template <class T, std::size_t N>
    inline void avx_int_batch<T, N>::store_aligned(T* dst) const
    {
        _mm256_store_si256((__m256i*) dst, m_value);
    }

    template <class T, std::size_t N>
    inline void avx_int_batch<T, N>::store_unaligned(T* dst) const
    {
        _mm256_storeu_si256((__m256i*) dst, m_value);
    }

    template <class T, std::size_t N>
    inline int avx_int_batch<T, N>::operator[](std::size_t index) const
    {
        alignas(32) T x[N];
        store_aligned(x);
        return x[index & (N - 1)];
    }

    namespace detail
    {
        template <class T>
        struct int8_batch_kernel
        {
            using batch_type = batch<T, 32>;
            using value_type = T;
            using batch_bool_type = batch_bool<T, 32>;

            constexpr static bool is_signed = std::is_signed<T>::value;

            static batch_type neg(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_sub_epi8(_mm256_setzero_si256(), rhs);
#else
                XSIMD_SPLIT_AVX(rhs);
                __m128i res_low = _mm_sub_epi8(_mm_setzero_si128(), rhs_low);
                __m128i res_high = _mm_sub_epi8(_mm_setzero_si128(), rhs_high);
                XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
            }

            static batch_type add(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_add_epi8(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_add_epi8, lhs, rhs);
#endif
            }

            static batch_type sub(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_sub_epi8(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_sub_epi8, lhs, rhs);
#endif
            }

            static batch_type mul(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                batch_type upper = _mm256_and_si256(_mm256_mullo_epi16(lhs, rhs), _mm256_srli_epi16(_mm256_set1_epi16(-1), 8));
                batch_type lower = _mm256_slli_epi16(_mm256_mullo_epi16(_mm256_srli_si256(lhs, 1), _mm256_srli_si256(rhs, 1)), 8);
                return _mm256_or_si256(upper, lower);
#else
                // Note implement with conversion to epi16
                XSIMD_MACRO_UNROLL_BINARY(*);
#endif
            }

            static batch_type div(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                auto to_float = [](__m256i val) {
                    // sign matters for conversion to epi32!
                    if (std::is_signed<T>::value)
                    {
                        return
                            _mm256_cvtepi32_ps(
                                _mm256_cvtepi8_epi32(
                                    _mm256_extractf128_si256(val, 0)
                                )
                            );
                    }
                    else
                    {
                        return
                            _mm256_cvtepi32_ps(
                                _mm256_cvtepu8_epi32(
                                    _mm256_extractf128_si256(val, 0)
                                )
                            );
                    }
                };

                auto to_int8 = [](__m256 x, __m256 y) {
                    auto v0 = _mm256_cvttps_epi32(x);
                    auto v1 = _mm256_cvttps_epi32(y);
                    // here the sign doesn't matter ... just an interpretation detail
                    auto a = _mm256_unpacklo_epi8(v0, v1);  // 08.. .... 19.. .... 4C.. .... 5D.. ....
                    auto b = _mm256_unpackhi_epi8(v0, v1);  // 2A.. .... 3B.. .... 6E.. .... 7F.. ....
                    auto c = _mm256_unpacklo_epi8(a, b);    // 028A .... .... .... 46CE ...
                    auto d = _mm256_unpackhi_epi8(a, b);    // 139B .... .... .... 57DF ...
                    auto e = _mm256_unpacklo_epi8(c, d);    // 0123 89AB .... .... 4567 CDEF ...
                    return _mm_unpacklo_epi32(_mm256_extractf128_si256(e, 0), 
                                              _mm256_extractf128_si256(e, 1));  // 0123 4567 89AB CDEF

                };

                auto insert = [](__m256i a, __m128i b)
                {
                    #if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                        return _mm256_inserti128_si256(a, b, 1);
                    #else
                        return _mm256_insertf128_si256(a, b, 1);
                    #endif
                };

                batch<float, 8> res_1 = _mm256_div_ps(to_float(lhs), to_float(rhs));
                batch<float, 8> res_2 = _mm256_div_ps(to_float(_mm256_permute4x64_epi64(lhs, 0x01)),
                                                      to_float(_mm256_permute4x64_epi64(rhs, 0x01)));
                batch<float, 8> res_3 = _mm256_div_ps(to_float(_mm256_permute4x64_epi64(lhs, 0x02)),
                                                      to_float(_mm256_permute4x64_epi64(rhs, 0x02)));
                batch<float, 8> res_4 = _mm256_div_ps(to_float(_mm256_permute4x64_epi64(lhs, 0x03)),
                                                      to_float(_mm256_permute4x64_epi64(rhs, 0x03)));

                return batch_type(
                    insert(_mm256_castsi128_si256(to_int8(res_1, res_2)),
                          to_int8(res_3, res_4))
                );
#else
                XSIMD_MACRO_UNROLL_BINARY(/);
#endif
            }

            static batch_type mod(const batch_type& lhs, const batch_type& rhs)
            {
                XSIMD_MACRO_UNROLL_BINARY(%);
            }

            static batch_bool_type eq(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_cmpeq_epi8(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_cmpeq_epi8, lhs, rhs);
#endif
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(lhs == rhs);
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                if (is_signed)
                {
                    return _mm256_cmpgt_epi8(rhs, lhs);
                }
                else
                {
                    auto xor_lhs = _mm256_xor_si256(lhs, _mm256_set1_epi8(std::numeric_limits<int8_t>::lowest()));
                    auto xor_rhs = _mm256_xor_si256(rhs, _mm256_set1_epi8(std::numeric_limits<int8_t>::lowest()));
                    return _mm256_cmpgt_epi8(xor_lhs, xor_rhs);
                }
#else
                if (is_signed)
                {
                    XSIMD_APPLY_SSE_FUNCTION(_mm_cmpgt_epi8, rhs, lhs);
                }
                else
                {
                    auto xor_lhs = _mm256_xor_si256(lhs, _mm256_set1_epi8(std::numeric_limits<int8_t>::lowest()));
                    auto xor_rhs = _mm256_xor_si256(rhs, _mm256_set1_epi8(std::numeric_limits<int8_t>::lowest()));
                    XSIMD_APPLY_SSE_FUNCTION(_mm_cmpgt_epi8, xor_lhs, xor_rhs);
                }
#endif
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(rhs < lhs);
            }

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_and_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_and_si128, lhs, rhs);
#endif
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_or_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_or_si128, lhs, rhs);
#endif
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_xor_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_xor_si128, lhs, rhs);
#endif
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_xor_si256(rhs, _mm256_set1_epi8(-1));
#else
                XSIMD_SPLIT_AVX(rhs);
                __m128i res_low = _mm_xor_si128(rhs_low, _mm_set1_epi8(-1));
                __m128i res_high = _mm_xor_si128(rhs_high, _mm_set1_epi8(-1));
                XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_andnot_si256(lhs, rhs);
#else
                XSIMD_APPLY_SSE_FUNCTION(_mm_andnot_si128, lhs, rhs);
#endif
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                if (is_signed)
                {
                    return _mm256_min_epi8(lhs, rhs);
                }
                else
                {
                    return _mm256_min_epu8(lhs, rhs);
                }
#else
                if (is_signed)
                {
                    XSIMD_APPLY_SSE_FUNCTION(_mm_min_epi8, lhs, rhs);
                }
                else
                {
                    XSIMD_APPLY_SSE_FUNCTION(_mm_min_epu8, lhs, rhs);
                }
#endif
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                if (is_signed)
                {
                    return _mm256_max_epi8(lhs, rhs);
                }
                else
                {
                    return _mm256_max_epu8(lhs, rhs);
                }
#else
                if (is_signed)
                {
                    XSIMD_APPLY_SSE_FUNCTION(_mm_max_epi8, lhs, rhs);
                }
                else
                {
                    XSIMD_APPLY_SSE_FUNCTION(_mm_max_epu8, lhs, rhs);
                }
#endif
            }

            static batch_type abs(const batch_type& rhs)
            {
                if (!is_signed)
                {
                    return rhs;
                }

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_sign_epi8(rhs, rhs);
#else
                XSIMD_SPLIT_AVX(rhs);
                __m128i res_low = _mm_sign_epi8(rhs_low, rhs_low);
                __m128i res_high = _mm_sign_epi8(rhs_high, rhs_high);
                XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
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

            // TODO use conversion to int16_t
            static value_type hadd(const batch_type& lhs)
            {
                alignas(32) value_type tmp_lhs[32];
                lhs.store_aligned(&tmp_lhs[0]);
                value_type res = 0;
                unroller<32>([&](std::size_t i) {
                    res += tmp_lhs[i];
                });
                return res;
            }

            static batch_type select(const batch_bool_type& cond, const batch_type& a, const batch_type& b)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION
                return _mm256_blendv_epi8(b, a, cond);
#else
                XSIMD_SPLIT_AVX(cond);
                XSIMD_SPLIT_AVX(a);
                XSIMD_SPLIT_AVX(b);
                __m128i res_low = _mm_blendv_epi8(b_low, a_low, cond_low);
                __m128i res_high = _mm_blendv_epi8(b_high, a_high, cond_high);
                XSIMD_RETURN_MERGED_SSE(res_low, res_high);
#endif
            }
        };

        template <>
        struct batch_kernel<int8_t, 32>
            : public int8_batch_kernel<int8_t>
        {
        };

        template <>
        struct batch_kernel<uint8_t, 32>
            : public int8_batch_kernel<uint8_t>
        {
        };
    }

    namespace avx_detail
    {
        template <class F, class T, std::size_t N>
        inline batch<T, N> shift_impl(F&& f, const batch<T, N>& lhs, int32_t rhs)
        {
            alignas(32) T tmp_lhs[N], tmp_res[N];
            lhs.store_aligned(&tmp_lhs[0]);
            unroller<N>([&](std::size_t i) {
                tmp_res[i] = f(tmp_lhs[i], rhs);
            });
            return batch<T, N>(tmp_res, aligned_mode());
        }
    }

    // TODO implement by converting to int16
    inline batch<int8_t, 32> operator<<(const batch<int8_t, 32>& lhs, int32_t rhs)
    {
        return avx_detail::shift_impl([](int8_t val, int32_t rhs) {
            return val << rhs;
        }, lhs, rhs);
    }

    inline batch<int8_t, 32> operator>>(const batch<int8_t, 32>& lhs, int32_t rhs)
    {
        return avx_detail::shift_impl([](int8_t val, int32_t rhs) {
            return val >> rhs;
        }, lhs, rhs);
    }

    inline batch<uint8_t, 32> operator<<(const batch<uint8_t, 32>& lhs, int32_t rhs)
    {
        return avx_detail::shift_impl([](uint8_t val, int32_t rhs) {
            return val << rhs;
        }, lhs, rhs);
    }

    inline batch<uint8_t, 32> operator>>(const batch<uint8_t, 32>& lhs, int32_t rhs)
    {
        return avx_detail::shift_impl([](uint8_t val, int32_t rhs) {
            return val >> rhs;
        }, lhs, rhs);
    }
}

#undef XSIMD_APPLY_SSE_FUNCTION
#undef XSIMD_RETURN_MERGED_SSE
#undef XSIMD_SPLIT_AVX

#endif
