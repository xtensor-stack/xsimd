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
        template <class... Args, class Enable = detail::is_array_initializer_t<bool, sizeof(MASK) * 8, Args...>>
        batch_bool_avx512(Args... args);
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
    template <class... Args, class>
    inline batch_bool_avx512<MASK, T>::batch_bool_avx512(Args... args)
        : batch_bool_avx512({{static_cast<bool>(args)...}})
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
            return (T(init[IX]) << IX) | get_init_value_impl<T, I...>(init);
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

    namespace detail
    {
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

        template <>
        struct mask_type<64>
        {
            using type = __mmask64;
        };

        template <class T, std::size_t N>
        struct batch_bool_kernel_avx512
        {
            using batch_type = batch_bool<T, N>;
            using mt = typename mask_type<N>::type;

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return mt(lhs) & mt(rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return mt(lhs) | mt(rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return mt(lhs) ^ mt(rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return ~mt(rhs);
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return mt(lhs) ^ mt(rhs);
            }

            static batch_type equal(const batch_type& lhs, const batch_type& rhs)
            {
                return (~mt(lhs)) ^ mt(rhs);
            }

            static batch_type not_equal(const batch_type& lhs, const batch_type& rhs)
            {
                return mt(lhs) ^ mt(rhs);
            }

            static bool all(const batch_type& rhs)
            {
                return mt(rhs) == mt(-1);
            }

            static bool any(const batch_type& rhs)
            {
                return mt(rhs) != mt(0);
            }
        };
    }
}

#endif