/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_INVTRIGO_HPP
#define XSIMD_INVTRIGO_HPP

#include "xsimd_fp_sign.hpp"
#include "xsimd_horner.hpp"
#include "xsimd_numerical_constant.hpp"

namespace xsimd
{
    namespace detail
    {
        template <class B, class T = typename B::value_type>
        struct invtrigo_kernel;

        /* origin: boost/simd/arch/common/detail/simd/f_invtrig.hpp */
        /*
         * ====================================================
         * copyright 2016 NumScale SAS
         *
         * Distributed under the Boost Software License, Version 1.0.
         * (See copy at http://boost.org/LICENSE_1_0.txt)
         * ====================================================
         */
        template <class B>
        struct invtrigo_kernel<B, float>
        {
            static inline B asin(const B& a)
            {
                B x = abs(a);
                B sign = bitofsign(a);
                auto x_larger_05 = x > B(0.5);
                B z = select(x_larger_05, B(0.5) * (B(1.) - x), x * x);
                x = select(x_larger_05, sqrt(z), x);
                B z1 = horner<B,
                    0x3e2aaae4,
                    0x3d9980f6,
                    0x3d3a3ec7,
                    0x3cc617e3,
                    0x3d2cb352
                >(z);
                z1 = fma(z1, z * x, x);
                z = select(x_larger_05, pio2<B>() - (z1 + z1), z1);
                return z ^ sign;
            }
        };

        /* origin: boost/simd/arch/common/detail/simd/d_invtrig.hpp */
        /*
         * ====================================================
         * copyright 2016 NumScale SAS
         *
         * Distributed under the Boost Software License, Version 1.0.
         * (See copy at http://boost.org/LICENSE_1_0.txt)
         * ====================================================
         */
        template <class B>
        struct invtrigo_kernel<B, double>
        {
            static inline B asin(const B& a)
            {
                B x = abs(a);
                auto small_cond = x < sqrteps<B>();
                B ct1 = B(detail::caster64_t(int64_t(0x3fe4000000000000)).f);
                B zz1 = B(1.) - x;
                B vp = zz1 * horner<B,
                    0x403c896240f3081dll,
                    0xc03991aaac01ab68ll,
                    0x401bdff5baf33e6all,
                    0xbfe2079259f9290fll,
                    0x3f684fc3988e9f08ll
                >(zz1) /
                    horner1<B,
                    0x40756709b0b644bell,
                    0xc077fe08959063eell,
                    0x40626219af6a7f42ll,
                    0xc035f2a2b6bf5d8cll
                    >(zz1);
                zz1 = sqrt(zz1 + zz1);
                B z = pio4<B>() - zz1;
                zz1 = fms(zz1, vp, pio_2lo<B>());
                z = z - zz1;
                zz1 = z + pio4<B>();
                B zz2 = a * a;
                z = zz2 * horner<B,
                    0xc020656c06ceafd5ll,
                    0x40339007da779259ll,
                    0xc0304331de27907bll,
                    0x4015c74b178a2dd9ll,
                    0xbfe34341333e5c16ll,
                    0x3f716b9b0bd48ad3ll
                >(zz2) /
                    horner1<B,
                    0xc04898220a3607acll,
                    0x4061705684ffbf9dll,
                    0xc06265bb6d3576d7ll,
                    0x40519fc025fe9054ll,
                    0xc02d7b590b5e0eabll
                    >(zz2);
                zz2 = fma(x, z, x);
                return select(x > B(1.), nan<B>(),
                    select(small_cond, x,
                        select(x > ct1, zz1, zz2)
                    ) ^ bitofsign(a));
            }
        };
    }
}

#endif
