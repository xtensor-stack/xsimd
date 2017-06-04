/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_NUMERICAL_CONSTANT_HPP
#define XSIMD_NUMERICAL_CONSTANT_HPP

#include <limits>
#include "../types/xsimd_types_include.hpp"

namespace xsimd
{
    template <class T>
    constexpr T infinity() noexcept;

    template <class T>
    constexpr T invlog_2() noexcept;

    template <class T>
    constexpr T invlog10_2() noexcept;

    template <class T>
    constexpr T log_2() noexcept;

    template <class T>
    constexpr T log_2hi() noexcept;

    template <class T>
    constexpr T log_2lo() noexcept;

    template <class T>
    constexpr T log10_2hi() noexcept;

    template <class T>
    constexpr T log10_2lo() noexcept;

    template <class T>
    constexpr T logeps() noexcept;

    template <class T>
    constexpr as_integer_t<T> maxexponent() noexcept;

    template <class T>
    constexpr T maxflint() noexcept;

    template <class T>
    constexpr T maxlog() noexcept;

    template <class T>
    constexpr T maxlog2() noexcept;

    template <class T>
    constexpr T maxlog10() noexcept;

    template <class T>
    constexpr T minlog() noexcept;

    template <class T>
    constexpr T minlog2() noexcept;

    template <class T>
    constexpr T minlog10() noexcept;

    template <class T>
    constexpr T minusinfinity() noexcept;

    template <class T>
    constexpr T minuszero() noexcept;

    template <class T>
    constexpr T nan() noexcept;

    template <class T>
    constexpr int32_t nmb() noexcept;

    template <class T>
    constexpr T smallestposval() noexcept;

    template <class T>
    constexpr T twotonmb() noexcept;

    /***************************
     * infinity implementation *
     ***************************/

    template <class T>
    constexpr T infinity() noexcept
    {
        return T(infinity<typename T::value_type>());
    }

    template <>
    constexpr float infinity<float>() noexcept
    {
        return std::numeric_limits<float>::infinity();
    }

    template <>
    constexpr double infinity<double>() noexcept
    {
        return std::numeric_limits<double>::infinity();
    }
    
    /***************************
     * invlog_2 implementation *
     ***************************/

    template <class T>
    constexpr T invlog_2() noexcept
    {
        return T(invlog_2<typename T::value_type>());
    }

    template <>
    constexpr float invlog_2<float>() noexcept
    {
        return float(1.442695040888963407359924681001892137426645954152986);
    }

    template <>
    constexpr double invlog_2<double>() noexcept
    {
        return double(1.442695040888963407359924681001892137426645954152986);
    }

    /*****************************
     * invlog10_2 implementation *
     *****************************/

    template <class T>
    constexpr T invlog10_2() noexcept
    {
        return T(invlog10_2<typename T::value_type>());
    }

    template <>
    constexpr float invlog10_2<float>() noexcept
    {
        return float(3.32192809488736234787031942949);
    }

    template <>
    constexpr double invlog10_2<double>() noexcept
    {
        return double(3.32192809488736234787031942949);
    }

    /************************
     * log_2 implementation *
     ************************/

    template <class T>
    constexpr T log_2() noexcept
    {
        return T(log_2<typename T::value_type>());
    }

    template <>
    constexpr float log_2<float>() noexcept
    {
        return float(0.6931471805599453094172321214581765680755001343602553);
    }

    template <>
    constexpr double log_2<double>() noexcept
    {
        return double(0.6931471805599453094172321214581765680755001343602553);
    }

    /**************************
     * log_2hi implementation *
     **************************/

    template <class T>
    constexpr T log_2hi() noexcept
    {
        return T(log_2hi<typename T::value_type>());
    }

    template <>
    constexpr float log_2hi<float>() noexcept
    {
        return detail::caster32_t(0x3f318000U).f;
    }

    template <>
    constexpr double log_2hi<double>() noexcept
    {
        return detail::caster64_t(uint64_t(0x3fe62e42fee00000U)).f;
    }

    /**************************
     * log_2lo implementation *
     **************************/

    template <class T>
    constexpr T log_2lo() noexcept
    {
        return T(log_2lo<typename T::value_type>());
    }

    template <>
    constexpr float log_2lo<float>() noexcept
    {
        return detail::caster32_t(0xb95e8083U).f;
    }

    template <>
    constexpr double log_2lo<double>() noexcept
    {
        return detail::caster64_t(uint64_t(0x3dea39ef35793c76U)).f;
    }

    /****************************
     * log10_2hi implementation *
     ****************************/

    template <class T>
    constexpr T log10_2hi() noexcept
    {
        return T(log10_2hi<typename T::value_type>());
    }

    template <>
    constexpr float log10_2hi<float>() noexcept
    {
        return detail::caster32_t(0x3e9a0000U).f;
    }

    template <>
    constexpr double log10_2hi<double>() noexcept
    {
        return detail::caster64_t(uint64_t(0x3fd3440000000000U)).f;
    }

    /****************************
     * log10_2lo implementation *
     ****************************/

    template <class T>
    constexpr T log10_2lo() noexcept
    {
        return T(log10_2lo<typename T::value_type>());
    }

    template <>
    constexpr float log10_2lo<float>() noexcept
    {
        return detail::caster32_t(0x39826a14U).f;
    }

    template <>
    constexpr double log10_2lo<double>() noexcept
    {
        return detail::caster64_t(uint64_t(0x3ed3509f79fef312U)).f;
    }

    /*************************
     * logeps implementation *
     *************************/

    template <class T>
    constexpr T logeps() noexcept
    {
        return T(logeps<typename T::value_type>());
    }

    template <>
    constexpr float logeps<float>() noexcept
    {
        return detail::caster32_t(0xc17f1402U).f;
    }

    template <>
    constexpr double logeps<double>() noexcept
    {
        return detail::caster64_t(uint64_t(0xc04205966f2b4f12U)).f;
    }

    /******************************
     * maxexponent implementation *
     ******************************/

    template <class T>
    constexpr as_integer_t<T> maxexponent() noexcept
    {
        return as_integer_t<T>(maxexponent<typename T::value_type>());
    }

    template<>
    constexpr int32_t maxexponent<float>() noexcept
    {
        return 127;
    }

    template <>
    constexpr int64_t maxexponent<double>() noexcept
    {
        return 1023;
    }

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

    /*************************
     * maxlog implementation *
     *************************/

    template <class T>
    constexpr T maxlog() noexcept
    {
        return T(maxlog<typename T::value_type>());
    }
    
    template <>
    constexpr float maxlog<float>() noexcept
    {
        return 88.3762626647949f;
    }

    template <>
    constexpr double maxlog<double>() noexcept
    {
        return 709.78271289338400;
    }

    /**************************
     * maxlog2 implementation *
     **************************/

    template <class T>
    constexpr T maxlog2() noexcept
    {
        return T(maxlog2<typename T::value_type>());
    }

    template <>
    constexpr float maxlog2<float>() noexcept
    {
        return 127.0f;
    }

    template <>
    constexpr double maxlog2<double>() noexcept
    {
        return 1023.0;
    }

    /***************************
     * maxlog10 implementation *
     ***************************/

    template <class T>
    constexpr T maxlog10() noexcept
    {
        return T(maxlog10<typename T::value_type>());
    }

    template <>
    constexpr float maxlog10<float>() noexcept
    {
        return 38.23080825805664f;
    }

    template <>
    constexpr double maxlog10<double>() noexcept
    {
        return 308.2547155599167;
    }

    /*************************
     * minlog implementation *
     *************************/

    template <class T>
    constexpr T minlog() noexcept
    {
        return T(minlog<typename T::value_type>());
    }

    template <>
    constexpr float minlog<float>() noexcept
    {
        return -88.3762626647949f;
    }

    template <>
    constexpr double minlog<double>() noexcept
    {
        return -708.3964185322641;
    }

    /**************************
     * minlog2 implementation *
     **************************/

    template <class T>
    constexpr T minlog2() noexcept
    {
        return T(minlog2<typename T::value_type>());
    }

    template <>
    constexpr float minlog2<float>() noexcept
    {
        return -127.0f;
    }

    template <>
    constexpr double minlog2<double>() noexcept
    {
        return -1023.0;
    }

    /***************************
     * minlog10 implementation *
     ***************************/

    template <class T>
    constexpr T minlog10() noexcept
    {
        return T(minlog10<typename T::value_type>());
    }

    template <>
    constexpr float minlog10<float>() noexcept
    {
        return -37.89999771118164f;
    }

    template <>
    constexpr double minlog10<double>() noexcept
    {
        return -308.2547155599167;
    }

    /********************************
     * minusinfinity implementation *
     ********************************/

    template <class T>
    constexpr T minusinfinity() noexcept
    {
        return T(minusinfinity<typename T::value_type>());
    }

    template <>
    constexpr float minusinfinity<float>() noexcept
    {
        return -infinity<float>();
    }

    template <>
    constexpr double minusinfinity<double>() noexcept
    {
        return -infinity<double>();
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
     * nan implementation *
     **********************/

    template <class T>
    constexpr T nan() noexcept
    {
        return T(nan<typename T::value_type>());
    }

    template <>
    constexpr float nan<float>() noexcept
    {
        return detail::caster32_t(0xFFFFFFFFU).f;
    }

    template <>
    constexpr double nan<double>() noexcept
    {
        return detail::caster64_t(uint64_t(0xFFFFFFFFFFFFFFFFU)).f;
    }

    /**********************
     * nmb implementation *
     **********************/

    template <class T>
    constexpr int32_t nmb() noexcept
    {
        return nmb<typename T::value_type>();
    }

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

    /*********************************
     * smallestposval implementation *
     *********************************/

    template <class T>
    constexpr T smallestposval() noexcept
    {
        return T(smallestposval<typename T::value_type>());
    }

    template <>
    constexpr float smallestposval<float>() noexcept
    {
        return 1.1754944e-38f;
    }

    template <>
    constexpr double smallestposval<double>() noexcept
    {
        return 2.225073858507201e-308;
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
