/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_NUMERICAL_CONSTANT_HPP
#define XSIMD_NUMERICAL_CONSTANT_HPP

#include "../types/xsimd_types_include.hpp"

namespace xsimd
{
    template <class T>
    constexpr T maxflint() noexcept;

    template <class T>
    constexpr int32_t maxexponent() noexcept;

    template <class T>
    constexpr T minuszero() noexcept;

    template <class T>
    constexpr int32_t nmb() noexcept;

    template <class T>
    constexpr T twotonmb() noexcept;

    /***************************
     * maxflint implementation *
     ***************************/

    template <class T>
    constexpr T maxflint() noexcept
    {
        return T(maxflint<typename T::value_type>());
    }

    template <>
    constexpr float maxflint<float>() noexcept
    {
        return 16777216.0f;
    }

    template <>
    constexpr double maxflint<double>() noexcept
    {
        return 9007199254740992.0;
    }

    /******************************
     * maxexponent implementation *
     ******************************/

    template<>
    constexpr int32_t maxexponent<float>() noexcept
    {
        return 127;
    }

    template <>
    constexpr int32_t maxexponent<double>() noexcept
    {
        return 1023;
    }

    /****************************
     * minuszero implementation *
     ****************************/

    template <class T>
    constexpr T minuszero() noexcept
    {
        return T(minuszero<typename T::value_type>());
    }

    template <>
    constexpr float minuszero<float>() noexcept
    {
        return -0.0f;
    }

    template <>
    constexpr double minuszero<double>() noexcept
    {
        return -0.0;
    }

    /**********************
     * nmb implementation *
     **********************/

    template <>
    constexpr int32_t nmb<float>() noexcept
    {
        return 23;
    }

    template <>
    constexpr int32_t nmb<double>() noexcept
    {
        return 52;
    }

    /***************************
     * twotonmb implementation *
     ***************************/

    template <class T>
    constexpr T twotonmb() noexcept
    {
        return T(twotonmb<typename T::value_type>());
    }

    template <>
    constexpr float twotonmb<float>() noexcept
    {
        return 8388608.0f;
    }

    template <>
    constexpr double twotonmb<double>() noexcept
    {
        return 4503599627370496.0;
    }

}

#endif

