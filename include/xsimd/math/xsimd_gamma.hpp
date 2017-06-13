/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_GAMMA_HPP
#define XSIMD_GAMMA_HPP

#include "xsimd_exponential.hpp"
#include "xsimd_horner.hpp"
#include "xsimd_logarithm.hpp"
#include "xsimd_trigonometric.hpp"

namespace xsimd
{

    template <class T, std::size_t N>
    batch<T, N> tgamma(const batch<T, N>& x);

    /*************************
    * tgamma implementation *
    *************************/

    namespace detail
    {
        /* origin: boost/simd/arch/common/detail/generic/stirling_kernel.hpp */
        /*
         * ====================================================
         * copyright 2016 NumScale SAS
         *
         * Distributed under the Boost Software License, Version 1.0.
         * (See copy at http://boost.org/LICENSE_1_0.txt)
         * ====================================================
         */

        template <class B, class T = typename B::value_type>
        struct stirling_kernel;

        template <class B>
        struct stirling_kernel<B, float>
        {
            static inline B compute(const B& x)
            {
                return horner<B,
                    0x3daaaaab,
                    0x3b638e39,
                    0xbb2fb930,
                    0xb970b359
                >(x);
            }

            static inline B split_limit()
            {
                return B(detail::caster32_t(uint32_t(0x41d628f6)).f);
            }

            static inline B large_limit()
            {
                return B(detail::caster32_t(uint32_t(0x420c28f3)).f);
            }
        };

        template <class B>
        struct stirling_kernel<B, double>
        {
            static inline B compute(const B& x)
            {
                return horner<B,
                    0x3fb5555555555986ll, //   8.33333333333482257126E-2
                    0x3f6c71c71b98c5fdll, //   3.47222221605458667310E-3
                    0xbf65f72607d44fd7ll, //  -2.68132617805781232825E-3
                    0xbf2e166b27e61d7cll, //  -2.29549961613378126380E-4
                    0x3f49cc72592d7293ll  //   7.87311395793093628397E-4
                >(x);
            }

            static inline B split_limit()
            {
                return B(detail::caster64_t(uint64_t(0x4061e083ba3443d4)).f);
            }

            static inline B large_limit()
            {
                return B(detail::caster64_t(uint64_t(0x4065800000000000)).f);
            }
        };

        /* origin: boost/simd/arch/common/simd/function/stirling.hpp */
        /*
         * ====================================================
         * copyright 2016 NumScale SAS
         *
         * Distributed under the Boost Software License, Version 1.0.
         * (See copy at http://boost.org/LICENSE_1_0.txt)
         * ====================================================
         */

        template <class B>
        inline B stirling(const B& a)
        {
            const B stirlingsplitlim = stirling_kernel<B>::split_limit();
            const B stirlinglargelim = stirling_kernel<B>::large_limit();
            B x = select(a >= B(0.), a, nan<B>());
            B w = B(1.) / x;
            w = fma(w, stirling_kernel<B>::compute(w), B(1.));
            B y = exp(-x);
            auto test = (x < stirlingsplitlim);
            B z = x - B(0.5);
            z = select(test, z, B(0.5) * z);
            B v = exp(z * log(abs(x)));
            y *= v;
            y = select(test, y, y * v);
            y *= sqrt_2pi<B>() * w;
#ifndef XSIMD_NO_INFINITIES
            y = select(isinf(x), x, y);
#endif
            return select(x > stirlinglargelim, infinity<B>(), y);
        }
    }

    /* origin: boost/simd/arch/common/detail/generic/gamma_kernel.hpp */
    /*
     * ====================================================
     * copyright 2016 NumScale SAS
     *
     * Distributed under the Boost Software License, Version 1.0.
     * (See copy at http://boost.org/LICENSE_1_0.txt)
     * ====================================================
     */

    namespace detail
    {
        template <class B, class T = typename B::value_type>
        struct gamma_kernel;

        template <class B>
        struct gamma_kernel<B, float>
        {
            static inline B compute(const B& x)
            {
                return horner<B,
                    0x3f800000UL, //  9.999999757445841E-01
                    0x3ed87799UL, //  4.227874605370421E-01
                    0x3ed2d411UL, //  4.117741948434743E-01
                    0x3da82a34UL, //  8.211174403261340E-02
                    0x3d93ae7cUL, //  7.211014349068177E-02
                    0x3b91db14UL, //  4.451165155708328E-03
                    0x3ba90c99UL, //  5.158972571345137E-03
                    0x3ad28b22UL  //  1.606319369134976E-03
                >(x);
            }
        };

        template <class B>
        struct gamma_kernel<B, double>
        {
            static inline B compute(const B& x)
            {
                return horner<B,
                    0x3ff0000000000000ULL, // 9.99999999999999996796E-1
                    0x3fdfa1373993e312ULL, // 4.94214826801497100753E-1
                    0x3fca8da9dcae7d31ULL, // 2.07448227648435975150E-1
                    0x3fa863d918c423d3ULL, // 4.76367800457137231464E-2
                    0x3f8557cde9db14b0ULL, // 1.04213797561761569935E-2
                    0x3f5384e3e686bfabULL, // 1.19135147006586384913E-3
                    0x3f24fcb839982153ULL  // 1.60119522476751861407E-4
                >(x) /
                    horner<B,
                    0x3ff0000000000000ULL, //  1.00000000000000000320E00
                    0x3fb24944c9cd3c51ULL, //  7.14304917030273074085E-2
                    0xbfce071a9d4287c2ULL, // -2.34591795718243348568E-1
                    0x3fa25779e33fde67ULL, //  3.58236398605498653373E-2
                    0x3f8831ed5b1bb117ULL, //  1.18139785222060435552E-2
                    0xBf7240e4e750b44aULL, // -4.45641913851797240494E-3
                    0x3f41ae8a29152573ULL, //  5.39605580493303397842E-4
                    0xbef8487a8400d3aFULL  // -2.31581873324120129819E-5
                    >(x);
            }
        };
    }

    /* origin: boost/simd/arch/common/simd/function/gamma.hpp */
    /*
     * ====================================================
     * copyright 2016 NumScale SAS
     *
     * Distributed under the Boost Software License, Version 1.0.
     * (See copy at http://boost.org/LICENSE_1_0.txt)
     * ====================================================
     */

    namespace detail
    {
        template <class B>
        B tgamma_large_negative(const B& a)
        {
            B st = stirling(a);
            B p = floor(a);
            B sgngam = select(is_even(p), -B(1.), B(1.));
            B z = a - p;
            auto test2 = z < B(0.5);
            z = select(test2, z - B(1.), z);
            z = a * sin_impl(z, trigo_pi_tag());
            z = abs(z);
            return sgngam * pi<B>() / (z * st);
        }

        template <class B, class BB>
        B tgamma_other(const B& a, const BB& test)
        {
            B x = select(test, B(2.), a);
#ifndef XSIMD_NO_INFINITIES
            auto inf_result = (a == infinity<B>());
            x = select(inf_result, B(2.), x);
#endif
            B z = B(1.);
            auto test1 = (x >= B(3.));
            while (any(test1))
            {
                x = select(test1, x - B(1.), x);
                z = select(test1, z * x, z);
                test1 = (x >= B(3.));
            }
            test1 = (x < B(0.));
            while (any(test1))
            {
                z = select(test1, z / x, z);
                x = select(test1, x + B(1.), x);
                test1 = (x < B(0.));
            }
            auto test2 = (x < B(2.));
            while (any(test2))
            {
                z = select(test2, z / x, z);
                x = select(test2, x + B(1.), x);
                test2 = (x < B(2.));
            }
            x = z * gamma_kernel<B>::compute(x - B(2.));
#ifndef XSIMD_NO_INFINITIES
            return select(inf_result, a, x);
#else
            return x;
#endif
        }

        template <class B>
        inline B tgamma_impl(const B& a)
        {
            auto nan_result = (a < B(0.) && is_flint(a));
#ifndef XSIMD_NO_INVALIDS
            nan_result = is_nan(a) || nan_result;
#endif
            B q = abs(a);
            auto test = (a < B(-33.));
            B r = nan<B>();
            if (any(test))
            {
                r = tgamma_large_negative(q);
                if (all(test))
                    return select(nan_result, nan<B>(), r);
            }
            B r1 = tgamma_other(a, test);
            B r2 = select(test, r, r1);
            return select(a == B(0.), copysign(infinity<B>(), a), select(nan_result, nan<B>(), r2));
        }
    }

    template <class T, std::size_t N>
    inline batch<T, N> tgamma(const batch<T, N>& x)
    {
        return detail::tgamma_impl(x);
    }
}

#endif
