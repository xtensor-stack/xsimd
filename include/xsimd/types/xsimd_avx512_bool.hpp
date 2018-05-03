/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_BOOL_HPP
#define XSIMD_AVX512_BOOL_HPP

#include "xsimd_utils.hpp"

#include "xsimd_base.hpp"

namespace xsimd
{
    template <class MASK, class T>
    class batch_bool_avx512;

    template <class MASK, class T>
    class batch_bool_avx512
    {
    public:

        batch_bool_avx512();
        explicit batch_bool_avx512(bool b);
        batch_bool_avx512(const bool (&init)[sizeof(MASK) * 8]);

        batch_bool_avx512(const MASK& rhs);
        batch_bool_avx512& operator=(const __m512& rhs);

        bool operator[](std::size_t index) const;

        operator MASK() const;

    private:

        MASK m_value;
    };

    template <class MASK, class T>
    inline batch_bool_avx512<MASK, T>::batch_bool_avx512()
    {
    }

    template <class MASK, class T>
    inline batch_bool_avx512<MASK, T>::batch_bool_avx512(bool b)
        : m_value(b ? -1 : 0)
    {
    }

    namespace detail
    {
        template <class T>
        constexpr T get_init_value_impl(const bool (&/*init*/)[sizeof(T) * 8])
        {
            return T(0);
        }

        template <class T, std::size_t IX, std::size_t... I>
        constexpr T get_init_value_impl(const bool (&init)[sizeof(T) * 8])
        {
            return (init[IX] << IX) | get_init_value_impl<T, I...>(init);
        }
        
        template <class T, std::size_t... I>
        constexpr T get_init_value(const bool (&init)[sizeof(T) * 8], detail::index_sequence<I...>)
        {
            return get_init_value_impl<T, I...>(init);
        }
    }

    template <class MASK, class T>
    inline batch_bool_avx512<MASK, T>::batch_bool_avx512(const bool (&init)[sizeof(MASK) * 8])
        : m_value(detail::get_init_value<MASK>(init, detail::make_index_sequence<sizeof(MASK) * 8>{}))
    {
    }

    template <class MASK, class T>
    inline batch_bool_avx512<MASK, T>::batch_bool_avx512(const MASK& rhs)
        : m_value(rhs)
    {
    }

    template <class MASK, class T>
    inline batch_bool_avx512<MASK, T>::operator MASK() const
    {
        return m_value;
    }

    template <class MASK, class T>
    inline bool batch_bool_avx512<MASK, T>::operator[](std::size_t idx) const
    {
        return (m_value & (1 << idx)) != 0;
    }

    template <std::size_t N> 
    struct mask_type;

    template <>
    struct mask_type<8>
    {
        using type = __mmask8;
    };

    template <>
    struct mask_type<16>
    {
        using type = __mmask16;
    };

#define AVX512_BOOL_OPERATOR(T, N, OP, CNT)                                                        \
    inline batch_bool<T, N> OP (const batch_bool<T, N>& lhs, const batch_bool<T, N>& rhs)          \
    {                                                                                              \
        using mt = typename mask_type<N>::type;                                                    \
        return CNT;                                                                                \
    }                                                                                              \


#define AVX512_BOOL_UNARY_OPERATOR(T, N, OP, CNT)                                                  \
    inline batch_bool<T, N> OP (const batch_bool<T, N>& rhs)                                       \
    {                                                                                              \
        using mt = typename mask_type<N>::type;                                                    \
        return CNT;                                                                                \
    }                                                                                              \


#define GENERATE_AVX512_BOOL_OPS(T, N)                                 \
    AVX512_BOOL_OPERATOR(T, N, operator==, (~mt(lhs)) ^ mt(rhs));      \
    AVX512_BOOL_OPERATOR(T, N, operator!=, mt(lhs) ^ mt(rhs));         \
    AVX512_BOOL_OPERATOR(T, N, operator&, mt(lhs) & mt(rhs));          \
    AVX512_BOOL_OPERATOR(T, N, operator|, mt(lhs) | mt(rhs));          \
    AVX512_BOOL_OPERATOR(T, N, operator^, mt(lhs) ^ mt(rhs));          \
    AVX512_BOOL_OPERATOR(T, N, bitwise_andnot, mt(lhs) ^ mt(rhs));     \
    AVX512_BOOL_UNARY_OPERATOR(T, N, operator~, ~mt(rhs));             \
    AVX512_BOOL_UNARY_OPERATOR(T, N, all, mt(rhs) == mt(-1));          \
    AVX512_BOOL_UNARY_OPERATOR(T, N, any, mt(rhs) != mt(0));           \

}

#endif