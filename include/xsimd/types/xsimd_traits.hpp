/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_TRAITS_HPP
#define XSIMD_TRAITS_HPP

#include "../config/xsimd_include.hpp"
#if defined(XSIMD_USE_AVX)
#include "xavx_float.hpp"
#include "xavx_double.hpp"
#elif defined(XSIMD_USE_SSE)
#include "xsse_float.hpp"
#include "xsse_double.hpp"
#endif

namespace xsimd
{

    template <class T>
    struct simd_traits
    {
        using type = T;
        static constexpr size_t size = 1;
    };

    template <class T>
    struct revert_simd_traits
    {
        using type = T;
        static constexpr size_t size = simd_traits<type>::size;
    };

    template <class T>
    using simd_type = typename simd_traits<T>::type;

    template <class T>
    using revert_simd_type = typename revert_simd_traits<T>::type;

#ifdef XSIMD_USE_AVX

    template <>
    struct simd_traits<float>
    {
        using type = vector8f;
        static constexpr size_t size = 8;
    };

    template <>
    struct revert_simd_traits<vector8f>
    {
        using type = float;
        static constexpr size_t size = simd_traits<type>::size;
    };

    template <>
    struct simd_traits<double>
    {
        using type = vector4d;
        static constexpr size_t size = 4;
    };

    template <>
    struct revert_simd_traits<vector4d>
    {
        using type = double;
        static constexpr size_t size = simd_traits<type>::size;
    };

#elif defined(XSIMD_USE_SSE)

    template <>
    struct simd_traits<float>
    {
        using type = vector4f;
        static constexpr size_t size = 4;
    };

    template <>
    struct revert_simd_traits<vector4f>
    {
        using type = float;
        static constexpr size_t size = simd_traits<type>::size;
    };

    template <>
    struct simd_traits<double>
    {
        using type = vector2d;
        static constexpr size_t size = 2;
    };

    template <>
    struct revert_simd_traits<vector2d>
    {
        using type = double;
        static constexpr size_t size = simd_traits<type>::size;
    };

#endif

}

#endif

