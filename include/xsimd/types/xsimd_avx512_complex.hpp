/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_AVX512_COMPLEX_HPP
#define XSIMD_AVX512_COMPLEX_HPP

#include <complex>

#ifdef XSIMD_ENABLE_XTL_COMPLEX
#include "xtl/xcomplex.hpp"
#endif

#include "xsimd_avx512_float.hpp"
#include "xsimd_avx512_double.hpp"
#include "xsimd_complex_base.hpp"

namespace xsimd
{

    /***************************************
     * batch_bool<std::complex<float>, 16> *
     ***************************************/

    template <>
    struct simd_batch_traits<batch_bool<std::complex<float>, 16>>
        : complex_batch_bool_traits<std::complex<float>, float, 16, 64>
    {
    };

    template<>
    class batch_bool<std::complex<float>, 16>
        : public simd_complex_batch_bool<batch_bool<std::complex<float>, 16>>
    {
    public:

        batch_bool() = default;
        using simd_complex_batch_bool::simd_complex_batch_bool;

        using real_batch = batch_bool<float, 16>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch_bool(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7,
                   bool b8, bool b9, bool b10, bool b11, bool b12, bool b13, bool b14, bool b15)
            : simd_complex_batch_bool(real_batch(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15))
        {
        }
    };

    /**********************************
     * batch<std::complex<float>, 16> *
     **********************************/

    template <>
    struct simd_batch_traits<batch<std::complex<float>, 16>>
        : complex_batch_traits<std::complex<float>, float, 16, 64>
    {
    };

    template <>
    class batch<std::complex<float>, 16>
        : public simd_complex_batch<batch<std::complex<float>, 16>>
    {
    public:

        batch() = default;
        using simd_complex_batch::simd_complex_batch;

        using value_type = std::complex<float>;
        using real_batch = batch<float, 16>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch(value_type c0, value_type c1, value_type c2, value_type c3,
              value_type c4, value_type c5, value_type c6, value_type c7,
              value_type c8, value_type c9, value_type c10, value_type c11,
              value_type c12, value_type c13, value_type c14, value_type c15)
            : simd_complex_batch(real_batch(c0.real(), c1.real(), c2.real(), c3.real(), c4.real(), c5.real(), c6.real(), c7.real(),
                                            c8.real(), c9.real(), c10.real(), c11.real(), c12.real(), c13.real(), c14.real(), c15.real()),
                                 real_batch(c0.imag(), c1.imag(), c2.imag(), c3.imag(), c4.imag(), c5.imag(), c6.imag(), c7.imag(),
                                            c8.imag(), c9.imag(), c10.imag(), c11.imag(), c12.imag(), c13.imag(), c14.imag(), c15.imag()))
        {
        }
    };

    /***************************************
     * batch_bool<std::complex<double>, 8> *
     ***************************************/

    template <>
    struct simd_batch_traits<batch_bool<std::complex<double>, 8>>
        : complex_batch_bool_traits<std::complex<double>, double, 8, 64>
    {
    };

    template<>
    class batch_bool<std::complex<double>, 8>
        : public simd_complex_batch_bool<batch_bool<std::complex<double>, 8>>
    {
    public:

        batch_bool() = default;
        using simd_complex_batch_bool::simd_complex_batch_bool;

        using real_batch = batch_bool<double, 8>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch_bool(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7)
            : simd_complex_batch_bool(real_batch(b0, b1, b2, b3, b4, b5, b6, b7))
        {
        }
    };

    /**********************************
     * batch<std::complex<double>, 8> *
     **********************************/

    template <>
    struct simd_batch_traits<batch<std::complex<double>, 8>>
        : complex_batch_traits<std::complex<double>, double, 8, 64>
    {
    };

    template <>
    class batch<std::complex<double>, 8>
        : public simd_complex_batch<batch<std::complex<double>, 8>>
    {
    public:

        batch() = default;
        using simd_complex_batch::simd_complex_batch;

        using value_type = std::complex<double>;
        using real_batch = batch<double, 8>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch(value_type c0, value_type c1, value_type c2, value_type c3,
              value_type c4, value_type c5, value_type c6, value_type c7)
            : simd_complex_batch(real_batch(c0.real(), c1.real(), c2.real(), c3.real(), c4.real(), c5.real(), c6.real(), c7.real()),
                                 real_batch(c0.imag(), c1.imag(), c2.imag(), c3.imag(), c4.imag(), c5.imag(), c6.imag(), c7.imag()))
        {
        }
    };

#ifdef XSIMD_ENABLE_XTL_COMPLEX

    /*****************************************************
     * batch_bool<xtl::xcomplex<float, float, i3ec>, 16> *
     *****************************************************/

    template <bool i3ec>
    struct simd_batch_traits<batch_bool<xtl::xcomplex<float, float, i3ec>, 16>>
        : complex_batch_bool_traits<xtl::xcomplex<float, float, i3ec>, float, 16, 64>
    {
    };

    template<bool i3ec>
    class batch_bool<xtl::xcomplex<float, float, i3ec>, 16>
        : public simd_complex_batch_bool<batch_bool<xtl::xcomplex<float, float, i3ec>, 16>>
    {
    public:

        batch_bool() = default;
        using simd_complex_batch_bool::simd_complex_batch_bool;

        using real_batch = batch_bool<float, 16>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch_bool(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7,
                   bool b8, bool b9, bool b10, bool b11, bool b12, bool b13, bool b14, bool b15)
            : simd_complex_batch_bool(real_batch(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15))
        {
        }
    };

    /***********************************************
     * batch<xtl::xcomplex<float, float, i3ec>, 16> *
     ***********************************************/

    template <bool i3ec>
    struct simd_batch_traits<batch<xtl::xcomplex<float, float, i3ec>, 16>>
        : complex_batch_traits<xtl::xcomplex<float, float, i3ec>, float, 16, 64>
    {
    };

    template <bool i3ec>
    class batch<xtl::xcomplex<float, float, i3ec>, 8>
        : public simd_complex_batch<batch<xtl::xcomplex<float, float, i3ec>, 16>>
    {
    public:

        batch() = default;
        using simd_complex_batch::simd_complex_batch;

        using value_type = xtl::xcomplex<float, float, i3ec>;
        using real_batch = batch<float, 16>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch(value_type c0, value_type c1, value_type c2, value_type c3,
            value_type c4, value_type c5, value_type c6, value_type c7,
            value_type c8, value_type c9, value_type c10, value_type c11,
            value_type c12, value_type c13, value_type c14, value_type c15)
            : simd_complex_batch(real_batch(c0.real(), c1.real(), c2.real(), c3.real(), c4.real(), c5.real(), c6.real(), c7.real(),
                                            c8.real(), c9.real(), c10.real(), c11.real(), c12.real(), c13.real(), c14.real(), c15.real()),
                                 real_batch(c0.imag(), c1.imag(), c2.imag(), c3.imag(), c4.imag(), c5.imag(), c6.imag(), c7.imag(),
                                            c8.imag(), c9.imag(), c10.imag(), c11.imag(), c12.imag(), c13.imag(), c14.imag(), c15.imag()))
        {
        }
    };

    /******************************************************
     * batch_bool<xtl::xcomplex<double, double, i3ec>, 8> *
     ******************************************************/

    template <bool i3ec>
    struct simd_batch_traits<batch_bool<xtl::xcomplex<double, double, i3ec>, 8>>
        : complex_batch_bool_traits<xtl::xcomplex<double, double, i3ec>, double, 8, 64>
    {
    };

    template<bool i3ec>
    class batch_bool<xtl::xcomplex<double, double, i3ec>, 8>
        : public simd_complex_batch_bool<batch_bool<xtl::xcomplex<double, double, i3ec>, 8>>
    {
    public:

        batch_bool() = default;
        using simd_complex_batch_bool::simd_complex_batch_bool;

        using real_batch = batch_bool<double, 4>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch_bool(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7)
            : simd_complex_batch_bool(real_batch(b0, b1, b2, b3, b4, b5, b6, b7))
        {
        }
    };

    /*************************************************
     * batch<xtl::xcomplex<double, double, i3ec>, 8> *
     *************************************************/

    template <bool i3ec>
    struct simd_batch_traits<batch<xtl::xcomplex<double, double, i3ec>, 8>>
        : complex_batch_traits<xtl::xcomplex<double, double, i3ec>, double, 8, 64>
    {
    };

    template <bool i3ec>
    class batch<xtl::xcomplex<double, double, i3ec>, 8>
        : public simd_complex_batch<batch<xtl::xcomplex<double, double, i3ec>, 8>>
    {
    public:

        batch() = default;
        using simd_complex_batch::simd_complex_batch;

        using value_type = xtl::xcomplex<double, double, i3ec>;
        using real_batch = batch<double, 2>;
        // VS2015 has a bug with inheriting constructors involving SFINAE
        batch(value_type c0, value_type c1, value_type c2, value_type c3,
              value_type c4, value_type c5, value_type c6, value_type c7)
            : simd_complex_batch(real_batch(c0.real(), c1.real(), c2.real(), c3.real(), c4.real(), c5.real(), c6.real(), c7.real()),
                                 real_batch(c0.imag(), c1.imag(), c2.imag(), c3.imag(), c4.imag(), c5.imag(), c6.imag(), c7.imag()))
        {
        }
    };

#endif
}

#endif
