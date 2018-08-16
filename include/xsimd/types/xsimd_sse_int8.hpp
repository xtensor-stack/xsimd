/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_SSE_INT8_HPP
#define XSIMD_SSE_INT8_HPP

#include <cstdint>

#include "xsimd_base.hpp"

namespace xsimd
{

    /********************
     * batch_bool<T, N> *
     ********************/

    template <>
    struct simd_batch_traits<batch_bool<int8_t, 16>>
    {
        using value_type = int8_t;
        static constexpr std::size_t size = 16;
        using batch_type = batch<int8_t, 16>;
        static constexpr std::size_t align = 16;
    };

    template <>
    struct simd_batch_traits<batch_bool<uint8_t, 16>>
    {
        using value_type = uint8_t;
        static constexpr std::size_t size = 16;
        using batch_type = batch<uint8_t, 16>;
        static constexpr std::size_t align = 16;
    };

    template <class T, std::size_t N>
    class sse_batch_bool : public simd_batch_bool<batch_bool<T, N>>
    {
    public:

        sse_batch_bool();
        explicit sse_batch_bool(bool b);
        template <class... Args, class Enable = detail::is_array_initializer_t<bool, N, Args...>>
        sse_batch_bool(Args... args);
        sse_batch_bool(const __m128i& rhs);
        sse_batch_bool& operator=(const __m128i& rhs);

        operator __m128i() const;

        bool operator[](std::size_t index) const;

    private:

        __m128i m_value;
    };

    template <>
    class batch_bool<int8_t, 16> : public sse_batch_bool<int8_t, 16>
    {
    public:
        using sse_batch_bool::sse_batch_bool;
    };

    template <>
    class batch_bool<uint8_t, 16> : public sse_batch_bool<uint8_t, 16>
    {
    public:
        using sse_batch_bool::sse_batch_bool;
    };

    /***********************
     * sse_int_batch<T, N> *
     ***********************/

    template <>
    struct simd_batch_traits<batch<int8_t, 16>>
    {
        using value_type = int8_t;
        static constexpr std::size_t size = 16;
        using batch_bool_type = batch_bool<int8_t, 16>;
        static constexpr std::size_t align = 16;
    };

    template <>
    struct simd_batch_traits<batch<uint8_t, 16>>
    {
        using value_type = uint8_t;
        static constexpr std::size_t size = 16;
        using batch_bool_type = batch_bool<uint8_t, 16>;
        static constexpr std::size_t align = 16;
    };

    template <class T, std::size_t N>
    class sse_int_batch : public simd_batch<batch<T, N>>
    {
    public:

        sse_int_batch();
        explicit sse_int_batch(T i);
        template <class... Args, class Enable = detail::is_array_initializer_t<T, N, Args...>>
        sse_int_batch(Args... args);
        explicit sse_int_batch(const T* src);
        sse_int_batch(const T* src, aligned_mode);
        sse_int_batch(const T* src, unaligned_mode);
        sse_int_batch(const __m128i& rhs);
        sse_int_batch& operator=(const __m128i& rhs);

        operator __m128i() const;

        batch<T, N>& load_aligned(const T* src);
        batch<T, N>& load_unaligned(const T* src);

        void store_aligned(T* dst) const;
        void store_unaligned(T* dst) const;

        T operator[](std::size_t index) const;

    private:

        __m128i m_value;
    };

    template <>
    class batch<int8_t, 16> : public sse_int_batch<int8_t, 16>
    {
    public:

        using base_class = sse_int_batch;
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
    class batch<uint8_t, 16> : public sse_int_batch<uint8_t, 16>
    {
    public:
        using sse_int_batch::sse_int_batch;
    };

    // template <>
    // class batch<int16_t, 8> : public sse_int_batch<int16_t, 8>
    // {
    // public:
    //     using sse_int_batch::sse_int_batch;
    // };

    // template <>
    // class batch<uint16_t, 8> : public sse_int_batch<uint16_t, 8>
    // {
    // public:
    //     using sse_int_batch::sse_int_batch;
    // };

    batch<int8_t, 16> operator<<(const batch<int8_t, 16>& lhs, int32_t rhs);
    batch<int8_t, 16> operator>>(const batch<int8_t, 16>& lhs, int32_t rhs);
    batch<uint8_t, 16> operator<<(const batch<uint8_t, 16>& lhs, int32_t rhs);
    batch<uint8_t, 16> operator>>(const batch<uint8_t, 16>& lhs, int32_t rhs);

    namespace sse_detail
    {
        template <class... Args>
        inline __m128i int_init(std::integral_constant<std::size_t, 1>, Args... args)
        {
            return _mm_setr_epi8(args...);
        }

        template <class... Args>
        inline __m128i int_init(std::integral_constant<std::size_t, 2>, Args... args)
        {
            return _mm_setr_epi16(args...);
        }
    }

    /***********************************
     * batch_bool<T, N> implementation *
     ***********************************/

    template <class T, std::size_t N>
    inline sse_batch_bool<T, N>::sse_batch_bool()
    {
    }

    template <class T, std::size_t N>
    inline sse_batch_bool<T, N>::sse_batch_bool(bool b)
        : m_value(_mm_set1_epi32(-(T)b))
    {
    }

    template <class T, std::size_t N>
    template <class... Args, class>
    inline sse_batch_bool<T, N>::sse_batch_bool(Args... args)
        : m_value(sse_detail::int_init(std::integral_constant<std::size_t, sizeof(T)>{}, -static_cast<T>(static_cast<bool>(args))...))
    {
    }

    template <class T, std::size_t N>
    inline sse_batch_bool<T, N>::sse_batch_bool(const __m128i& rhs)
        : m_value(rhs)
    {
    }

    template <class T, std::size_t N>
    inline sse_batch_bool<T, N>& sse_batch_bool<T, N>::operator=(const __m128i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    template <class T, std::size_t N>
    inline sse_batch_bool<T, N>::operator __m128i() const
    {
        return m_value;
    }

    template <class T, std::size_t N>
    inline bool sse_batch_bool<T, N>::operator[](std::size_t index) const
    {
        alignas(16) T x[N];
        _mm_store_si128((__m128i*)x, m_value);
        return static_cast<bool>(x[index & (N - 1)]);
    }

    namespace detail
    {
        template <class T>
        struct sse_int8_batch_bool_kernel
        {
            using batch_type = batch_bool<T, 16>;
            constexpr static bool is_signed = std::is_signed<T>::value;
            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_and_si128(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_or_si128(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_xor_si128(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return _mm_xor_si128(rhs, _mm_set1_epi32(-1));
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_andnot_si128(lhs, rhs);
            }

            static batch_type equal(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_cmpeq_epi8(lhs, rhs);
            }

            static batch_type not_equal(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(lhs == rhs);
            }

            static bool all(const batch_type& rhs)
            {
                return _mm_movemask_epi8(rhs) == 0xFFFF;
            }

            static bool any(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
                return !_mm_testz_si128(rhs, rhs);
#else
                return _mm_movemask_epi8(rhs) != 0;
#endif
            }
        };

        template <>
        struct batch_bool_kernel<int8_t, 16>
            : public sse_int8_batch_bool_kernel<int8_t>
        {
        };

        template <>
        struct batch_bool_kernel<uint8_t, 16>
            : public sse_int8_batch_bool_kernel<uint8_t>
        {
        };
    }

    /**************************************
     * sse_int_batch<T, N> implementation *
     **************************************/

    template <class T, std::size_t N>
    inline sse_int_batch<T, N>::sse_int_batch()
    {
    }

    template <class T, std::size_t N>
    inline sse_int_batch<T, N>::sse_int_batch(T i)
        : m_value(_mm_set1_epi8(i))
    {
    }

    template <class T, std::size_t N>
    template <class... Args, class>
    inline sse_int_batch<T, N>::sse_int_batch(Args... args)
        : m_value(sse_detail::int_init(args...))
    {
    }

    template <class T, std::size_t N>
    inline sse_int_batch<T, N>::sse_int_batch(const T* src)
        : m_value(_mm_loadu_si128((__m128i const*)src))
    {
    }

    template <class T, std::size_t N>
    inline sse_int_batch<T, N>::sse_int_batch(const T* src, aligned_mode)
        : m_value(_mm_load_si128((__m128i const*)src))
    {
    }

    template <class T, std::size_t N>
    inline sse_int_batch<T, N>::sse_int_batch(const T* src, unaligned_mode)
        : m_value(_mm_loadu_si128((__m128i const*)src))
    {
    }

    template <class T, std::size_t N>
    inline sse_int_batch<T, N>::sse_int_batch(const __m128i& rhs)
        : m_value(rhs)
    {
    }

    template <class T, std::size_t N>
    inline sse_int_batch<T, N>& sse_int_batch<T, N>::operator=(const __m128i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    template <class T, std::size_t N>
    inline sse_int_batch<T, N>::operator __m128i() const
    {
        return m_value;
    }

    template <class T, std::size_t N>
    inline batch<T, N>& sse_int_batch<T, N>::load_aligned(const T* src)
    {
        m_value = _mm_load_si128((__m128i const*)src);
        return (*this)();
    }

    template <class T, std::size_t N>
    inline batch<T, N>& sse_int_batch<T, N>::load_unaligned(const T* src)
    {
        m_value = _mm_loadu_si128((__m128i const*)src);
        return (*this)();
    }

    template <class T, std::size_t N>
    inline void sse_int_batch<T, N>::store_aligned(T* dst) const
    {
        _mm_store_si128((__m128i*)dst, m_value);
    }

    template <class T, std::size_t N>
    inline void sse_int_batch<T, N>::store_unaligned(T* dst) const
    {
        _mm_storeu_si128((__m128i*)dst, m_value);
    }

    template <class T, std::size_t N>
    inline T sse_int_batch<T, N>::operator[](std::size_t index) const
    {
        alignas(16) T x[N];
        store_aligned(x);
        return x[index & (N - 1)];
    }

    namespace detail
    {
        template <class T>
        struct sse_int8_batch_kernel
        {
            using batch_type = batch<T, 16>;
            using value_type = T;
            using batch_bool_type = batch_bool<T, 16>;

            static constexpr bool is_signed = std::is_signed<value_type>::value;

            static batch_type neg(const batch_type& rhs)
            {
                return _mm_sub_epi8(_mm_setzero_si128(), rhs);
            }

            static batch_type add(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_add_epi8(lhs, rhs);
            }

            static batch_type sub(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_sub_epi8(lhs, rhs);
            }

            static batch_type mul(const batch_type& lhs, const batch_type& rhs)
            {
                return sse_int8_batch_kernel::bitwise_or(
                    sse_int8_batch_kernel::bitwise_and(_mm_mullo_epi16(lhs, rhs), _mm_srli_epi16(_mm_set1_epi32(-1), 8)),
                    _mm_slli_epi16(_mm_mullo_epi16(_mm_srli_si128(lhs, 1), _mm_srli_si128(rhs, 1)), 8)
                );
            }

            static batch_type div(const batch_type& lhs, const batch_type& rhs)
            {
                // TODO implement divison as floating point!
                XSIMD_MACRO_UNROLL_BINARY(/);
            }

            static batch_bool_type eq(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_cmpeq_epi8(lhs, rhs);
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(lhs == rhs);
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(rhs < lhs);
            }

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_and_si128(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_or_si128(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_xor_si128(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return _mm_xor_si128(rhs, _mm_set1_epi8(-1));
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_andnot_si128(lhs, rhs);
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
                // TODO implement with hadd_epi16
                alignas(16) T tmp[16];
                rhs.store_aligned(tmp);
                T res = 0;
                for (int i = 0; i < 16; ++i)
                {
                    res += tmp[i];
                }
                return res;
            }

            static batch_type select(const batch_bool_type& cond, const batch_type& a, const batch_type& b)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
                return _mm_blendv_epi8(b, a, cond);
#else
                return _mm_or_si128(_mm_and_si128(cond, a), _mm_andnot_si128(cond, b));
#endif
            }
        };

        template <>
        struct batch_kernel<int8_t, 16>
            : public sse_int8_batch_kernel<int8_t>
        {
            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_cmplt_epi8(lhs, rhs);
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
                return _mm_min_epi8(lhs, rhs);
#else
                __m128i greater = _mm_cmpgt_epi8(lhs, rhs);
                return select(greater, rhs, lhs);
#endif
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
                return _mm_max_epi8(lhs, rhs);
#else
                __m128i greater = _mm_cmpgt_epi8(lhs, rhs);
                return select(greater, lhs, rhs);
#endif
            }


            static batch_type abs(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSSE3_VERSION
                return _mm_sign_epi8(rhs, rhs);
#else
                __m128i neg = _mm_sub_epi8(_mm_setzero_si128(), rhs);
                return _mm_min_epu8(rhs, neg);
#endif
            }
        };

        template <>
        struct batch_kernel<uint8_t, 16>
            : public sse_int8_batch_kernel<uint8_t>
        {
            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_cmplt_epi8(_mm_xor_si128(lhs, _mm_set1_epi8(std::numeric_limits<int8_t>::min())),
                                      _mm_xor_si128(rhs, _mm_set1_epi8(std::numeric_limits<int8_t>::min())));
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
                    return _mm_min_epu8(lhs, rhs);
#else
                    return select(lhs < rhs, lhs, rhs);
#endif
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
                    return _mm_max_epu8(lhs, rhs);
#else
                    return select(lhs < rhs, rhs, lhs);
#endif
            }

            static batch_type abs(const batch_type& rhs)
            {
                return rhs;
            }
        };
    }

    namespace sse_detail
    {
        template <class F, class T, std::size_t N>
        inline batch<T, N> shift_impl(F&& f, const batch<T, N>& lhs, int32_t rhs)
        {
            alignas(16) T tmp_lhs[N], tmp_res[N];
            lhs.store_aligned(&tmp_lhs[0]);
            unroller<N>([&](std::size_t i) {
                tmp_res[i] = f(tmp_lhs[i], rhs);
            });
            return batch<T, N>(tmp_res, aligned_mode());
        }
    }

    inline batch<int8_t, 16> operator<<(const batch<int8_t, 16>& lhs, int32_t rhs)
    {
        return sse_detail::shift_impl([](int8_t lhs, int32_t s) { return lhs << s; }, lhs, rhs);
    }

    inline batch<int8_t, 16> operator>>(const batch<int8_t, 16>& lhs, int32_t rhs)
    {
        return sse_detail::shift_impl([](int8_t lhs, int32_t s) { return lhs >> s; }, lhs, rhs);
    }

    inline batch<uint8_t, 16> operator<<(const batch<uint8_t, 16>& lhs, int32_t rhs)
    {
        return sse_detail::shift_impl([](uint8_t lhs, int32_t s) { return lhs << s; }, lhs, rhs);
    }

    inline batch<uint8_t, 16> operator>>(const batch<uint8_t, 16>& lhs, int32_t rhs)
    {
        return sse_detail::shift_impl([](uint8_t lhs, int32_t s) { return lhs >> s; }, lhs, rhs);
    }
}

#endif
