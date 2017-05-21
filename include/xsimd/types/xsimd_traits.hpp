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

#undef XSIMD_BATCH_INT_SIZE
#undef XSIMD_BATCH_FLOAT_SIZE
#undef XSIMD_BATCH_DOUBLE_SIZE

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
#include "xsimd_sse_int.hpp"
#include "xsimd_sse_float.hpp"
#include "xsimd_sse_double.hpp"
#define XSIMD_BATCH_INT_SIZE 4
#define XSIMD_BATCH_FLOAT_SIZE 4
#define XSIMD_BATCH_DOUBLE_SIZE 2
#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
#include "xsimd_avx_float.hpp"
#include "xsimd_avx_double.hpp"
#define XSIMD_BATCH_FLOAT_SIZE 8
#define XSIMD_BATCH_DOUBLE_SIZE 4
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

#ifdef XSIMD_BATCH_DOUBLE_SIZE

    template <>
    struct simd_traits<int>
    {
        using type = batch<int, XSIMD_BATCH_INT_SIZE>;
        static constexpr size_t size = type::size;
    };

    template <>
    struct revert_simd_traits<batch<int, XSIMD_BATCH_INT_SIZE>>
    {
        using type = int;
        static constexpr size_t size = simd_traits<type>::size;
    };

    template <>
    struct simd_traits<float>
    {
        using type = batch<float, XSIMD_BATCH_FLOAT_SIZE>;
        static constexpr size_t size = type::size;
    };

    template <>
    struct revert_simd_traits<batch<float, XSIMD_BATCH_FLOAT_SIZE>>
    {
        using type = float;
        static constexpr size_t size = simd_traits<type>::size;
    };

    template <>
    struct simd_traits<double>
    {
        using type = batch<double, XSIMD_BATCH_DOUBLE_SIZE>;
        static constexpr size_t size = type::size;
    };

    template <>
    struct revert_simd_traits<batch<double, XSIMD_BATCH_DOUBLE_SIZE>>
    {
        using type = double;
        static constexpr size_t size = simd_traits<type>::size;
    };

#endif

}

#endif

