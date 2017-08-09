/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_UTILS_HPP
#define XSIMD_UTILS_HPP

#include <cstdint>

namespace xsimd
{

    template <class T, size_t N>
    class batch;

    /**************
     * as_integer *
     **************/

    template <class T>
    struct as_integer;

    template <>
    struct as_integer<float>
    {
        using type = int32_t;
    };

    template <>
    struct as_integer<double>
    {
        using type = int64_t;
    };

    template <class T, std::size_t N>
    struct as_integer<batch<T, N>>
    {
        using type = batch<typename as_integer<T>::type, N>;
    };

    template <class T>
    using as_integer_t = typename as_integer<T>::type;

    /***********************
     * as_unsigned_integer *
     ***********************/

    template <class T>
    struct as_unsigned_integer;

    template <>
    struct as_unsigned_integer<float>
    {
        using type = uint32_t;
    };

    template <>
    struct as_unsigned_integer<double>
    {
        using type = uint64_t;
    };

    template <class T, std::size_t N>
    struct as_unsigned_integer<batch<T, N>>
    {
        using type = batch<typename as_unsigned_integer<T>::type, N>;
    };

    template <class T>
    using as_unsigned_integer_t = typename as_unsigned_integer<T>::type;

    /***********
     * as_float *
     ************/

    template <class T>
    struct as_float;

    template <>
    struct as_float<int32_t>
    {
        using type = float;
    };

    template <>
    struct as_float<int64_t>
    {
        using type = double;
    };

    template <class T, std::size_t N>
    struct as_float<batch<T, N>>
    {
        using type = batch<typename as_float<T>::type, N>;
    };

    template <class T>
    using as_float_t = typename as_float<T>::type;

    /**************
     * as_logical *
     **************/

    template <class T>
    struct as_logical;

    template <class T, std::size_t N>
    struct as_logical<batch<T, N>>
    {
        using type = batch_bool<T, N>;
    };

    template <class T>
    using as_logical_t = typename as_logical<T>::type;

    /********************
     * primitive caster *
     ********************/

    namespace detail
    {
        template <class UI, class I, class F>
        union generic_caster {
            UI ui;
            I i;
            F f;

            constexpr generic_caster(UI t)
                : ui(t) {}
            constexpr generic_caster(I t)
                : i(t) {}
            constexpr generic_caster(F t)
                : f(t) {}
        };

        using caster32_t = generic_caster<uint32_t, int32_t, float>;
        using caster64_t = generic_caster<uint64_t, int64_t, double>;

        template <class T>
        struct caster;

        template <>
        struct caster<float>
        {
            using type = caster32_t;
        };

        template <>
        struct caster<double>
        {
            using type = caster64_t;
        };

        template <class T>
        using caster_t = typename caster<T>::type;
    }
}

#endif
