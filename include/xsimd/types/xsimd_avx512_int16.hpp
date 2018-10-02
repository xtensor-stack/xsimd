/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_INT8_HPP
#define XSIMD_AVX512_INT8_HPP

#include "xsimd_avx512_int_base.hpp"

namespace xsimd
{   

    /***************************
     * batch_bool<int16_t, 32> *
     ***************************/

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

    /**********************
     * batch<int16_t, 32> *
     **********************/

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

}

