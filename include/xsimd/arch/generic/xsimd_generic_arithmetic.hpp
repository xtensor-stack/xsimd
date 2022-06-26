/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_GENERIC_ARITHMETIC_HPP
#define XSIMD_GENERIC_ARITHMETIC_HPP

#include <complex>
#include <type_traits>

#include "./xsimd_generic_details.hpp"

namespace xsimd
{

    namespace kernel
    {

        using namespace types;

        // bitwise_lshift
        template <class A, class T, class /*=typename std::enable_if<std::is_integral<T>::value, void>::type*/>
        inline batch<T, A> XSIMD_CALLCONV bitwise_lshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) noexcept
        {
            return detail::apply([](T x, T y) noexcept
                                 { return x << y; },
                                 self, other);
        }

        // bitwise_rshift
        template <class A, class T, class /*=typename std::enable_if<std::is_integral<T>::value, void>::type*/>
        inline batch<T, A> XSIMD_CALLCONV bitwise_rshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) noexcept
        {
            return detail::apply([](T x, T y) noexcept
                                 { return x >> y; },
                                 self, other);
        }

        // div
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> XSIMD_CALLCONV div(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) noexcept
        {
            return detail::apply([](T x, T y) noexcept -> T
                                 { return x / y; },
                                 self, other);
        }

        // fma
        template <class A, class T>
        inline batch<T, A> XSIMD_CALLCONV fma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires_arch<generic>) noexcept
        {
            return x * y + z;
        }

        template <class A, class T>
        inline batch<std::complex<T>, A> XSIMD_CALLCONV fma(batch<std::complex<T>, A> const& x, batch<std::complex<T>, A> const& y, batch<std::complex<T>, A> const& z, requires_arch<generic>) noexcept
        {
            auto res_r = fms(x.real(), y.real(), fms(x.imag(), y.imag(), z.real()));
            auto res_i = fma(x.real(), y.imag(), fma(x.imag(), y.real(), z.imag()));
            return { res_r, res_i };
        }

        // fms
        template <class A, class T>
        inline batch<T, A> XSIMD_CALLCONV fms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires_arch<generic>) noexcept
        {
            return x * y - z;
        }

        template <class A, class T>
        inline batch<std::complex<T>, A> XSIMD_CALLCONV fms(batch<std::complex<T>, A> const& x, batch<std::complex<T>, A> const& y, batch<std::complex<T>, A> const& z, requires_arch<generic>) noexcept
        {
            auto res_r = fms(x.real(), y.real(), fma(x.imag(), y.imag(), z.real()));
            auto res_i = fma(x.real(), y.imag(), fms(x.imag(), y.real(), z.imag()));
            return { res_r, res_i };
        }

        // fnma
        template <class A, class T>
        inline batch<T, A> XSIMD_CALLCONV fnma(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires_arch<generic>) noexcept
        {
            return -x * y + z;
        }

        template <class A, class T>
        inline batch<std::complex<T>, A> XSIMD_CALLCONV fnma(batch<std::complex<T>, A> const& x, batch<std::complex<T>, A> const& y, batch<std::complex<T>, A> const& z, requires_arch<generic>) noexcept
        {
            auto res_r = -fms(x.real(), y.real(), fma(x.imag(), y.imag(), z.real()));
            auto res_i = -fma(x.real(), y.imag(), fms(x.imag(), y.real(), z.imag()));
            return { res_r, res_i };
        }

        // fnms
        template <class A, class T>
        inline batch<T, A> XSIMD_CALLCONV fnms(batch<T, A> const& x, batch<T, A> const& y, batch<T, A> const& z, requires_arch<generic>) noexcept
        {
            return -x * y - z;
        }

        template <class A, class T>
        inline batch<std::complex<T>, A> XSIMD_CALLCONV fnms(batch<std::complex<T>, A> const& x, batch<std::complex<T>, A> const& y, batch<std::complex<T>, A> const& z, requires_arch<generic>) noexcept
        {
            auto res_r = -fms(x.real(), y.real(), fms(x.imag(), y.imag(), z.real()));
            auto res_i = -fma(x.real(), y.imag(), fma(x.imag(), y.real(), z.imag()));
            return { res_r, res_i };
        }

        // mul
        template <class A, class T, class /*=typename std::enable_if<std::is_integral<T>::value, void>::type*/>
        inline batch<T, A> XSIMD_CALLCONV mul(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) noexcept
        {
            return detail::apply([](T x, T y) noexcept -> T
                                 { return x * y; },
                                 self, other);
        }

    }

}

#endif
