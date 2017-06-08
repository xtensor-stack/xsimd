/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_HYPERBOLIC_HPP
#define XSIMD_HYPERBOLIC_HPP

#include <type_traits>
#include "xsimd_exponential.hpp"
#include "xsimd_fp_sign.hpp"

namespace xsimd
{

    template <class T, std::size_t N>
    batch<T, N> average(const batch<T, N>& x1, const batch<T, N>& x2);

    template <class T, std::size_t N>
    batch<T, N> sinh(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch<T, N> cosh(const batch<T, N>& x);

    template<class T, std::size_t N>
    batch<T, N> tanh(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch<T, N> asinh(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch<T, N> acosh(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch<T, N> atanh(const batch<T, N>& x);

    /***************************
     * average  implementation *
     ***************************/

    namespace detail
    {
        /* origin: boost/simd/arch/common/simd/function/average.hpp */
        /*
         * ====================================================
         * copyright 2016 NumScale SAS
         *
         * Distributed under the Boost Software License, Version 1.0.
         * (See copy at http://boost.org/LICENSE_1_0.txt)
         * ====================================================
         */

        template <class B, bool cond = std::is_floating_point<typename B::value_type>::value>
        struct average_impl
        {
            static inline B compute(const B& x1, const B& x2)
            {
                return (x1 & x2) + ((x1 ^ x2) >> 1);
            }
        };

        template <class B>
        struct average_impl<B, true>
        {
            static inline B compute(const B& x1, const B& x2)
            {
                return fma(x1, B(0.5), x2 * B(0.5));
            }
        };
    }

    template <class T, std::size_t N>
    inline batch<T, N> average(const batch<T, N>& x1, const batch<T, N>& x2)
    {
        return detail::average_impl<batch<T, N>>::compute(x1, x2);
    }

    /***********************
     * sinh implementation *
     ***********************/

    namespace detail
    {
        /* origin: boost/simd/arch/common/detail/generic/sinh_kernel.hpp */
        /*
         * ====================================================
         * copyright 2016 NumScale SAS
         *
         * Distributed under the Boost Software License, Version 1.0.
         * (See copy at http://boost.org/LICENSE_1_0.txt)
         * ====================================================
         */

        template <class B, class T = typename B::value_type>
        struct sinh_kernel;

        template <class B>
        struct sinh_kernel<B, float>
        {
            static inline B compute(const B& x)
            {
                B sqrx = x * x;
                return horner<B,
                    0x3f800000, // 1.0f
                    0x3e2aaacc, // 1.66667160211E-1f
                    0x3c087bbe, // 8.33028376239E-3f
                    0x39559e2f  // 2.03721912945E-4f
                >(sqrx) * x;
            }
        };

        template <class B>
        struct sinh_kernel<B, double>
        {
            static inline B compute(const B& x)
            {
                B sqrx = x * x;
                return fma(x, (horner<B,
                    0xc115782bdbf6ab05ull, //  -3.51754964808151394800E5
                    0xc0c694b8c71d6182ull, //  -1.15614435765005216044E4,
                    0xc064773a398ff4feull, //  -1.63725857525983828727E2,
                    0xbfe9435fe8bb3cd6ull  //  -7.89474443963537015605E-1
                >(sqrx) /
                    horner1<B,
                    0xc1401a20e4f90044ull, //  -2.11052978884890840399E6
                    0x40e1a7ba7ed72245ull, //   3.61578279834431989373E4,
                    0xc0715b6096e96484ull //  -2.77711081420602794433E2,
                    >(sqrx)) * sqrx, x);
            }
        };
    }

    /* origin: boost/simd/arch/common/simd/function/sinh.hpp */
    /*
     * ====================================================
     * copyright 2016 NumScale SAS
     *
     * Distributed under the Boost Software License, Version 1.0.
     * (See copy at http://boost.org/LICENSE_1_0.txt)
     * ====================================================
     */
    template <class T, std::size_t N>
    inline batch<T, N> sinh(const batch<T, N>& a)
    {
        using b_type = batch<T, N>;
        b_type half = b_type(0.5);
        b_type x = abs(a);
        auto lt1 = x < b_type(1.);
        b_type bts = bitofsign(a);
        b_type z(0.);
        if (any(lt1))
        {
            z = detail::sinh_kernel<b_type>::compute(x);
            if (all(lt1))
                return z ^ bts;
        }
        auto test1 = x > (maxlog<b_type>() - log_2<b_type>());
        b_type fac = select(test1, half, b_type(1.));
        b_type tmp = exp(x * fac);
        b_type tmp1 = half * tmp;
        b_type r = select(test1, tmp1 * tmp, tmp1 - half / tmp);
        return select(lt1, z, r) ^ bts;
    }

    /***********************
     * cosh implementation *
     ***********************/

     /* origin: boost/simd/arch/common/simd/function/cosh.hpp */
     /*
      * ====================================================
      * copyright 2016 NumScale SAS
      *
      * Distributed under the Boost Software License, Version 1.0.
      * (See copy at http://boost.org/LICENSE_1_0.txt)
      * ====================================================
      */

    template <class T, std::size_t N>
    inline batch<T, N> cosh(const batch<T, N>& a)
    {
        using b_type = batch<T, N>;
        b_type x = abs(a);
        auto test1 = x > (maxlog<b_type>() - log_2<b_type>());
        b_type fac = select(test1, b_type(0.5), b_type(1.));
        b_type tmp = exp(x * fac);
        b_type tmp1 = b_type(0.5) * tmp;
        return select(test1, tmp1 * tmp, average(tmp, b_type(1.) / tmp));
    }
}

#endif
