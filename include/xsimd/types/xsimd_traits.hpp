/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_TRAITS_HPP
#define XSIMD_TRAITS_HPP

#include <type_traits>

#include "xsimd_api.hpp"

namespace xsimd
{
    namespace detail
    {
        template <class T, class = void>
        struct has_batch : std::false_type
        {
        };

        template <class T>
        struct has_batch<T, check_size_t<sizeof(batch<T>)>> : std::true_type
        {
        };
    }

    template <class T, bool = detail::has_batch<T>::value>
    struct simd_traits
    {
        using type = T;
        using bool_type = bool;
        static constexpr size_t size = 1;
    };

    template <class T, bool B>
    constexpr size_t simd_traits<T, B>::size;

    template <class T>
    struct revert_simd_traits
    {
        using type = T;
        static constexpr size_t size = simd_traits<type>::size;
    };

    template <class T>
    constexpr size_t revert_simd_traits<T>::size;

    template <class T>
    struct simd_traits<T, true>
    {
        using type = batch<T>;
        using bool_type = typename type::batch_bool_type;
        static constexpr size_t size = type::size;
    };

    template <class T>
    struct simd_traits<std::complex<T>, true>
    {
        using type = batch<T>;
        //using bool_type = typename type::batch_bool_type;
        static constexpr size_t size = type::size;
    };




    template <class T>
    constexpr size_t simd_traits<T, true>::size;

    template <class T>
    struct revert_simd_traits<batch<T>>
    {
        using type = T;
        static constexpr size_t size = batch<T>::size;
    };

    template <class T>
    constexpr size_t revert_simd_traits<batch<T>>::size;

    template <class T>
    using simd_type = typename simd_traits<T>::type;

    template <class T>
    using simd_bool_type = typename simd_traits<T>::bool_type;

    template <class T>
    using revert_simd_type = typename revert_simd_traits<T>::type;

    /********************
     * simd_return_type *
     ********************/

    namespace detail
    {
        template <class T1, class T2>
        struct simd_condition
        {
            static constexpr bool value =
                (std::is_same<T1, T2>::value && !std::is_same<T1, bool>::value) ||
                (std::is_same<T1, bool>::value && !std::is_same<T2, bool>::value) ||
                std::is_same<T1, float>::value ||
                std::is_same<T1, double>::value ||
                std::is_same<T1, int8_t>::value ||
                std::is_same<T1, uint8_t>::value ||
                std::is_same<T1, int16_t>::value ||
                std::is_same<T1, uint16_t>::value ||
                std::is_same<T1, int32_t>::value ||
                std::is_same<T1, uint32_t>::value ||
                std::is_same<T1, int64_t>::value ||
                std::is_same<T1, uint64_t>::value ||
                std::is_same<T1, char>::value ||
                detail::is_complex<T1>::value;
        };

        template <class T1, class T2, class A>
        struct simd_return_type_impl
            : std::enable_if<simd_condition<T1, T2>::value, batch<T2, A>>
        {
        };
        template <class A>
        struct simd_return_type_impl<char, char, A>
            : std::conditional<std::is_signed<char>::value,
                               simd_return_type_impl<int8_t, int8_t, A>,
                               simd_return_type_impl<uint8_t, uint8_t, A>>::type
        {
        };

        template <class T2, class A>
        struct simd_return_type_impl<bool, T2, A>
            : std::enable_if<simd_condition<bool, T2>::value, batch_bool<T2, A>>
        {
        };

        template <class T1, class T2, class A>
        struct simd_return_type_impl<std::complex<T1>, T2, A>
            : std::enable_if<simd_condition<T1, T2>::value, batch<std::complex<T2>, A>>
        {
        };

        template <class T1, class T2, class A>
        struct simd_return_type_impl<std::complex<T1>, std::complex<T2>, A>
            : std::enable_if<simd_condition<T1, T2>::value, batch<std::complex<T2>, A>>
        {
        };
    }

    template <class T1, class T2, class A = default_arch>
    using simd_return_type = typename detail::simd_return_type_impl<T1, T2, A>::type;

    /************
     * is_batch *
     ************/

    template <class V>
    struct is_batch : std::false_type
    {
    };

    template <class T, class A>
    struct is_batch<batch<T, A>> : std::true_type
    {
    };

    /*****************
     * is_batch_bool *
     *****************/

    template <class V>
    struct is_batch_bool : std::false_type
    {
    };

    template <class T, class A>
    struct is_batch_bool<batch_bool<T, A>> : std::true_type
    {
    };

    /********************
     * is_batch_complex *
     ********************/

    template <class V>
    struct is_batch_complex : std::false_type
    {
    };

    template <class T, class A>
    struct is_batch_complex<batch<std::complex<T>, A>> : std::true_type
    {
    };
}

#endif
