//
// Copyright (c) 2016 Johan Mabille
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
//

#ifndef NX_SIMD_TRAITS_HPP
#define NX_SIMD_TRAITS_HPP

#if (NX_SSE_INSTR_SET > 6 && !NX_DISABLE_AVX)
#define NX_USE_AVX
#include "nx_avx_float.hpp"
#include "nx_avx_double.hpp"
#elif (NX_SSE_INSTR_SET > 0 && !NX_DISABLE_SSE)
#define NX_USE_SSE
#include "nx_sse_float.hpp"
#include "nx_sse_double.hpp"
#endif

namespace nxsimd
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

#ifdef NX_USE_AVX

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

#elif defined(NX_USE_SSE)

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

