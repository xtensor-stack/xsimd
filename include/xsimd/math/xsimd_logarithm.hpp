/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_LOGARITHM_HPP
#define XSIMD_LOGARITHM_HPP

#include "xsimd_numerical_constant.hpp"

namespace xsimd
{
    template <class T, std::size_t N>
    batch<T, N> log(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch<T, N> log2(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch<T, N> log10(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch<T, N> log1p(const batch<T, N>& x);

    /**********************
     * log implementation *
     **********************/

    namespace detail
    {
        template <class B, class T = typename B::value_type>
        struct log_kernel;

        template <class B>
        struct log_kernel<B, float>
        {
            static inline B compute(const B& a)
            {
                using i_type = as_integer_t<B>;
                B x = a;
                i_type k(0);
                auto isnez = (a != B(0.));
#ifndef XSIMD_NO_DENORMALS
                auto test = (a < smallestposval<B>()) && isnez;
                if (any(test))
                {
                    k = select(bool_cast(test), k - i_type(23), k);
                    x = select(test, x * B(8388608ul), x);
                }
#endif
                i_type ix = bitwise_cast<i_type>(x);
                ix += 0x3f800000 - 0x3f3504f3;
                k += (ix >> 23) - 0x7f;
                ix = (ix & i_type(0x007fffff)) + 0x3f3504f3;
                x = bitwise_cast<B>(ix);
                B f = --x;
                B s = f / (B(2.) + f);
                B z = s * s;
                B w = z * z;
                B t1 = w * horner<B, 0x3eccce13, 0x3e789e26>(w);
                B t2 = z * horner<B, 0x3f2aaaaa, 0x3e91e9ee>(w);
                B R = t2 + t1;
                B hfsq = B(0.5) * f * f;
                B dk = to_float(k);
                B r = fma(dk, log_2hi<B>(), fma(s, (hfsq + R), dk * log_2lo<B>()) - hfsq + f);
#ifndef XSIMD_NO_INIFINITIES
                B zz = select(isnez, select(a == infinity<B>(), infinity<B>(), r), minusinfinity<B>());
#else
                B zz = select(isnez, r, minusinfinity<B>());
#endif
                return select(!(a >= B(0.)), nan<B>(), zz);
            }
        };

        template <class B>
        struct log_kernel<B, double>
        {
            static inline B compute(const B& a)
            {
                using i_type = as_integer_t<B>;

                B x = a;
                i_type hx = bitwise_cast<i_type>(x) >> 32;
                i_type k(0);
                auto isnez = (a != B(0.));
#ifndef XSIMD_NO_DENORMALS
                auto test = (a < smallestposval<B>()) && isnez;
                if (any(test))
                {
                    k = select(bool_cast(test), k - i_type(54), k);
                    x = select(test, x * B(18014398509481984ull), x);
                }
#endif
                hx += 0x3ff00000 - 0x3fe6a09e;
                k += (hx >> 20) - 0x3ff;
                B dk = to_float(k);
                hx = (hx & i_type(0x000fffff)) + 0x3fe6a09e;
                x = bitwise_cast<B>(hx << 32 | (i_type(0xffffffff) & bitwise_cast<i_type>(x)));

                B f = --x;
                B hfsq = B(0.5) * f * f;
                B s = f / (B(2.) + f);
                B z = s * s;
                B w = z * z;

                B t1 = w * horner<B,
                    0x3fd999999997fa04ll,
                    0x3fcc71c51d8e78afll,
                    0x3fc39a09d078c69fll
                >(w);
                B t2 = z * horner<B,
                    0x3fe5555555555593ll,
                    0x3fd2492494229359ll,
                    0x3fc7466496cb03dell,
                    0x3fc2f112df3e5244ll
                >(w);
                B R = t2 + t1;
                B r = fma(dk, log_2hi<B>(), fma(s, (hfsq + R), dk * log_2lo<B>()) - hfsq + f);
#ifndef XSIMD_NO_INFINITIES
                B zz = select(isnez, select(a == infinity<B>(), infinity<B>(), r), minusinfinity<B>());
#else
                B zz = select(isnez, r, minusinfinity<B>());
#endif
                return select(!(a >= B(0.)), nan<B>(), zz);
            }
        };
    }

    template <class T, std::size_t N>
    inline batch<T, N> log(const batch<T, N>& x)
    {
        using b_type = batch<T, N>;
        return detail::log_kernel<b_type, T>::compute(x);
    }
}

#endif
