/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_SSE_COMPLEX_HPP
#define XSIMD_SSE_COMPLEX_HPP

#include <complex>

#ifdef XSIMD_ENABLE_XTL_COMPLEX
#include "xtl/xcomplex.hpp"
#endif

#include "xsimd_sse_float.hpp"
#include "xsimd_sse_double.hpp"
#include "xsimd_complex_base.hpp"

namespace xsimd
{

    /**************************************
     * batch_bool<std::complex<float>, 4> *
     **************************************/

    template <>
    struct simd_batch_traits<batch_bool<std::complex<float>, 4>>
        : complex_batch_bool_traits<std::complex<float>, float, 4, 16>
    {
    };

    template<>
    class batch_bool<std::complex<float>, 4>
        : public simd_complex_batch_bool<batch_bool<std::complex<float>, 4>>
    {
    public:

        batch_bool() = default;
        using simd_complex_batch_bool::simd_complex_batch_bool;

        using real_batch = batch_bool<float, 4>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch_bool(bool b0, bool b1, bool b2, bool b3)
            : simd_complex_batch_bool(real_batch(b0, b1, b2, b3))
        {
        }
    };

    /*********************************
     * batch<std::complex<float>, 4> *
     *********************************/

    template <>
    struct simd_batch_traits<batch<std::complex<float>, 4>>
        : complex_batch_traits<std::complex<float>, float, 4, 16>
    {
    };

    template <>
    class batch<std::complex<float>, 4>
        : public simd_complex_batch<batch<std::complex<float>, 4>>
    {
    public:

        batch() = default;
        using simd_complex_batch::simd_complex_batch;

        using value_type = std::complex<float>;
        using real_batch = batch<float, 4>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch(value_type c0, value_type c1, value_type c2, value_type c3)
            : simd_complex_batch(real_batch(c0.real(), c1.real(), c2.real(), c3.real()),
                                 real_batch(c0.imag(), c1.imag(), c2.imag(), c3.imag()))
        {
        }
    };

    /***************************************
     * batch_bool<std::complex<double>, 2> *
     ***************************************/

    template <>
    struct simd_batch_traits<batch_bool<std::complex<double>, 2>>
        : complex_batch_bool_traits<std::complex<double>, double, 2, 16>
    {
    };

    template<>
    class batch_bool<std::complex<double>, 2>
        : public simd_complex_batch_bool<batch_bool<std::complex<double>, 2>>
    {
    public:

        batch_bool() = default;
        using simd_complex_batch_bool::simd_complex_batch_bool;

        using real_batch = batch_bool<double, 2>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch_bool(bool b0, bool b1)
            : simd_complex_batch_bool(real_batch(b0, b1))
        {
        }
    };

    /**********************************
     * batch<std::complex<double>, 2> *
     **********************************/

    template <>
    struct simd_batch_traits<batch<std::complex<double>, 2>>
        : complex_batch_traits<std::complex<double>, double, 2, 16>
    {
    };

    template <>
    class batch<std::complex<double>, 2>
        : public simd_complex_batch<batch<std::complex<double>, 2>>
    {
    public:

        batch() = default;
        using simd_complex_batch::simd_complex_batch;

        using value_type = std::complex<double>;
        using real_batch = batch<double, 2>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch(value_type c0, value_type c1)
            : simd_complex_batch(real_batch(c0.real(), c1.real()),
                                 real_batch(c0.imag(), c1.imag()))
        {
        }
    };

#ifdef XSIMD_ENABLE_XTL_COMPLEX

    /****************************************************
     * batch_bool<xtl::xcomplex<float, float, i3ec>, 4> *
     ****************************************************/

    template <bool i3ec>
    struct simd_batch_traits<batch_bool<xtl::xcomplex<float, float, i3ec>, 4>>
        : complex_batch_bool_traits<xtl::xcomplex<float, float, i3ec>, float, 4, 16>
    {
    };

    template<bool i3ec>
    class batch_bool<xtl::xcomplex<float, float, i3ec>, 4>
        : public simd_complex_batch_bool<batch_bool<xtl::xcomplex<float, float, i3ec>, 4>>
    {
    public:

        batch_bool() = default;
        using simd_complex_batch_bool::simd_complex_batch_bool;

        using real_batch = batch_bool<float, 4>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch_bool(bool b0, bool b1, bool b2, bool b3)
            : simd_complex_batch_bool(real_batch(b0, b1, b2, b3))
        {
        }
    };

    /***********************************************
     * batch<xtl::xcomplex<float, float, i3ec>, 4> *
     ***********************************************/

    template <bool i3ec>
    struct simd_batch_traits<batch<xtl::xcomplex<float, float, i3ec>, 4>>
        : complex_batch_traits<xtl::xcomplex<float, float, i3ec>, float, 4, 16>
    {
    };

    template <bool i3ec>
    class batch<xtl::xcomplex<float, float, i3ec>, 4>
        : public simd_complex_batch<batch<xtl::xcomplex<float, float, i3ec>, 4>>
    {
    public:

        batch() = default;
        using simd_complex_batch::simd_complex_batch;

        using value_type = xtl::xcomplex<float, float, i3ec>;
        using real_batch = batch<float, 4>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch(value_type c0, value_type c1, value_type c2, value_type c3)
            : simd_complex_batch(real_batch(c0.real(), c1.real(), c2.real(), c3.real()),
                real_batch(c0.imag(), c1.imag(), c2.imag(), c3.imag()))
        {
        }
    };

    /******************************************************
     * batch_bool<xtl::xcomplex<double, double, i3ec>, 2> *
     ******************************************************/

    template <bool i3ec>
    struct simd_batch_traits<batch_bool<xtl::xcomplex<double, double, i3ec>, 2>>
        : complex_batch_bool_traits<xtl::xcomplex<double, double, i3ec>, double, 2, 16>
    {
    };

    template<bool i3ec>
    class batch_bool<xtl::xcomplex<double, double, i3ec>, 2>
        : public simd_complex_batch_bool<batch_bool<xtl::xcomplex<double, double, i3ec>, 2>>
    {
    public:

        batch_bool() = default;
        using simd_complex_batch_bool::simd_complex_batch_bool;

        using real_batch = batch_bool<double, 2>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch_bool(bool b0, bool b1)
            : simd_complex_batch_bool(real_batch(b0, b1))
        {
        }
    };

    /*************************************************
     * batch<xtl::xcomplex<double, double, i3ec>, 2> *
     *************************************************/

    template <bool i3ec>
    struct simd_batch_traits<batch<xtl::xcomplex<double, double, i3ec>, 2>>
        : complex_batch_traits<xtl::xcomplex<double, double, i3ec>, double, 2, 16>
    {
    };

    template <bool i3ec>
    class batch<xtl::xcomplex<double, double, i3ec>, 2>
        : public simd_complex_batch<batch<xtl::xcomplex<double, double, i3ec>, 2>>
    {
    public:

        batch() = default;
        using simd_complex_batch::simd_complex_batch;

        using value_type = xtl::xcomplex<double, double, i3ec>;
        using real_batch = batch<double, 2>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch(value_type c0, value_type c1)
            : simd_complex_batch(real_batch(c0.real(), c1.real()),
                real_batch(c0.imag(), c1.imag()))
        {
        }
    };

#endif

}

#endif
