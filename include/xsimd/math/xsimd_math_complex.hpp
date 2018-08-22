/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_MATH_COMPLEX_HPP
#define XSIMD_MATH_COMPLEX_HPP

#include "../types/xsimd_complex_base.hpp"
#include "xsimd_exponential.hpp"
#include "xsimd_hyperbolic.hpp"
#include "xsimd_logarithm.hpp"
#include "xsimd_power.hpp"
#include "xsimd_trigonometric.hpp"

namespace xsimd
{
    template <class X>
    typename X::real_batch abs(const simd_complex_batch<X>& z);

    template <class X>
    typename X::real_batch arg(const simd_complex_batch<X>& z);

    template <class X>
    X conj(const simd_complex_batch<X>& z);

    template <class X>
    X sqrt(const simd_complex_batch<X>& z);

    template <class X>
    typename simd_batch_traits<X>::real_value_type
    norm(const simd_complex_batch<X>& rhs);

    template <class X>
    X proj(const simd_complex_batch<X>& rhs);

    namespace detail
    {
        /*******
         * exp *
         *******/

        template <class T, std::size_t N>
        inline batch<T, N> exp_complex_impl(const batch<T, N>& z)
        {
            using b_type = batch<T, N>;
            using r_type = typename b_type::real_batch;
            r_type icos, isin;
            sincos(z.imag(), isin, icos);
            return exp(z.real()) * batch<T, N>(icos, isin);
        }

        template <class T, std::size_t N>
        struct exp_kernel<batch<std::complex<T>, N>, exp_tag, std::complex<T>>
        {
            using batch_type = batch<std::complex<T>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return exp_complex_impl(z);
            }
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, bool i3ec, std::size_t N>
        struct exp_kernel<batch<xtl::xcomplex<T, T, i3ec>, N>, exp_tag, xtl::xcomplex<T, T, i3ec>>
        {
            using batch_type = batch<xtl::xcomplex<T, T, i3ec>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return exp_complex_impl(z);
            }
        };
#endif

        /*******
         * log *
         *******/

        template <class T, std::size_t N>
        inline batch<T, N> log_complex_impl(const batch<T, N>& z)
        {
            return batch<T, N>(log(abs(z)), atan2(z.imag(), z.real()));
        }

        template <class T, std::size_t N>
        struct log_kernel<batch<std::complex<T>, N>, std::complex<T>>
        {
            using batch_type = batch<std::complex<T>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return log_complex_impl(z);
            }
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, bool i3ec, std::size_t N>
        struct log_kernel<batch<xtl::xcomplex<T, T, i3ec>, N>, xtl::xcomplex<T, T, i3ec>>
        {
            using batch_type = batch<xtl::xcomplex<T, T, i3ec>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return log_complex_impl(z);
            }
        };
#endif

        /*********
         * log10 *
         *********/

        template <class T, std::size_t N>
        inline batch<T, N> log10_complex_impl(const batch<T, N>& z)
        {
            using real_value_type = typename batch<T, N>::real_value_type;
            return log(z) / batch<T, N>(real_value_type(std::log(10)));
        }

        template <class T, std::size_t N>
        struct log10_kernel<batch<std::complex<T>, N>, std::complex<T>>
        {
            using batch_type = batch<std::complex<T>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return log10_complex_impl(z);
            }
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, bool i3ec, std::size_t N>
        struct log10_kernel<batch<xtl::xcomplex<T, T, i3ec>, N>, xtl::xcomplex<T, T, i3ec>>
        {
            using batch_type = batch<xtl::xcomplex<T, T, i3ec>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return log10_complex_impl(z);
            }
        };
#endif

        /*******
         * pow *
         *******/

        template <class T, std::size_t N>
        inline batch<T, N> pow_complex_impl(const batch<T, N>& a, const batch<T, N>& z)
        {
            using cplx_batch = batch<T, N>;
            using real_batch = typename cplx_batch::real_batch;
            real_batch absa = abs(a);
            real_batch arga = arg(a);
            real_batch x = z.real();
            real_batch y = z.imag();
            real_batch r = pow(absa, x);
            real_batch theta = x * arga;
            real_batch ze = zero<real_batch>();
            auto cond = (y == ze);
            r = select(cond, r, r * exp(-y * arga));
            theta = select(cond, theta, theta + y * log(absa));
            return select(absa == ze, cplx_batch(ze), cplx_batch(r * cos(theta), r * sin(theta)));
        }

        template <class T, std::size_t N>
        struct pow_kernel<batch<std::complex<T>, N>>
        {
            using batch_type = batch<std::complex<T>, N>;

            static inline batch_type compute(const batch_type& a, const batch_type& z)
            {
                return pow_complex_impl(a, z);
            }
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, bool i3ec, std::size_t N>
        struct pow_kernel<batch<xtl::xcomplex<T, T, i3ec>, N>>
        {
            using batch_type = batch<xtl::xcomplex<T, T, i3ec>, N>;

            static inline batch_type compute(const batch_type& a, const batch_type& z)
            {
                return pow_complex_impl(a, z);
            }
        };
#endif

        /*********
         * trigo *
         *********/

        template <class T, std::size_t N>
        inline void sincos_complex_impl(const batch<T, N>& z, batch<T, N>& si, batch<T, N>& co)
        {
            using b_type = batch<T, N>;
            using r_type = typename b_type::real_batch;
            r_type rcos = cos(z.real());
            r_type rsin = sin(z.real());
            r_type icosh = cosh(z.imag());
            r_type isinh = sinh(z.imag());
            si = b_type(rsin * icosh, rcos * isinh);
            co = b_type(rcos * icosh, -rsin * isinh);
        }

        template <class T, std::size_t N>
        inline batch<T, N> sin_complex_impl(const batch<T, N>& z)
        {
            return batch<T, N>(sin(z.real()) * cosh(z.imag()), cos(z.real()) * sinh(z.imag()));
        }

        template <class T, std::size_t N>
        inline batch<T, N> cos_complex_impl(const batch<T, N>& z)
        {
            return batch<T, N>(cos(z.real()) * cosh(z.imag()), -sin(z.real()) * sinh(z.imag()));
        }

        template <class T, std::size_t N>
        inline batch<T, N> tan_complex_impl(const batch<T, N>& z)
        {
            using b_type = batch<T, N>;
            using r_type = typename b_type::real_batch;
            r_type d = cos(2 * z.real()) + cosh(2 * z.imag());
            b_type winf(infinity<r_type>(), infinity<r_type>());
            r_type wreal = sin(2 * z.real()) / d;
            r_type wimag = sinh(2 * z.imag());
            b_type wres = select(isinf(wimag), b_type(wreal, r_type(1.)), b_type(wreal, wimag / d));
            return select(d == r_type(0.), winf, wres);
        }

        template <class T, std::size_t N>
        struct trigo_kernel<batch<std::complex<T>, N>>
        {
            using batch_type = batch<std::complex<T>, N>;

            static inline batch_type sin(const batch_type& z)
            {
                return sin_complex_impl(z);
            }

            static inline batch_type cos(const batch_type& z)
            {
                return cos_complex_impl(z);
            }

            static inline batch_type tan(const batch_type& z)
            {
                return tan_complex_impl(z);
            }

            static inline void sincos(const batch_type& z, batch_type& si, batch_type& co)
            {
                return sincos_complex_impl(z, si, co);
            }
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, bool i3ec, std::size_t N>
        struct trigo_kernel<batch<xtl::xcomplex<T, T, i3ec>, N>>
        {
            using batch_type = batch<xtl::xcomplex<T, T, i3ec>, N>;

            static inline batch_type sin(const batch_type& z)
            {
                return sin_complex_impl(z);
            }

            static inline batch_type cos(const batch_type& z)
            {
                return cos_complex_impl(z);
            }

            static inline batch_type tan(const batch_type& z)
            {
                return tan_complex_impl(z);
            }

            static inline void sincos(const batch_type& z, batch_type& si, batch_type& co)
            {
                return sincos_complex_impl(z, si, co);
            }
        };
#endif

        /************
         * invtrigo *
         ************/

        template <class T, std::size_t N>
        batch<T, N> asin_complex_impl(const batch<T, N>& z)
        {
            using b_type = batch<T, N>;
            using r_type = typename b_type::real_batch;

            r_type x = z.real();
            r_type y = z.imag();

            b_type ct(-y, x);
            b_type zz(r_type(1.) - (x - y) * (x + y), -2 * x * y);
            zz = log(ct + sqrt(zz));
            b_type resg(zz.imag(), -zz.real());

            return select(y == r_type(0.),
                          select(fabs(x) > r_type(1.),
                                 b_type(pio2<r_type>(), r_type(0.)),
                                 b_type(asin(x), r_type(0.))),
                          resg);
        }

        template <class T, std::size_t N>
        batch<T, N> acos_complex_impl(const batch<T, N>& z)
        {
            using b_type = batch<T, N>;
            using r_type = typename b_type::real_batch;
            b_type tmp = asin_complex_impl(z);
            return b_type(pio2<r_type>() - tmp.real(), -tmp.imag());
        }

        template <class T, std::size_t N>
        batch<T, N> atan_complex_impl(const batch<T, N>& z)
        {
            using b_type = batch<T, N>;
            using r_type = typename b_type::real_batch;
            r_type x = z.real();
            r_type y = z.imag();
            r_type x2 = x * x;
            r_type one = r_type(1.);
            r_type a = one - x2 - (y * y);
            r_type w = 0.5 * atan2(2. * x, a);
            r_type num = y + one;
            num = x2 + num * num;
            r_type den = y - one;
            den = x2 + den * den;
            b_type res(w, 0.25 * log(num / den));
            return res;
        }

        template <class T, std::size_t N>
        struct invtrigo_kernel<batch<std::complex<T>, N>>
        {
            using batch_type = batch<std::complex<T>, N>;

            static inline batch_type asin(const batch_type& z)
            {
                return asin_complex_impl(z);
            }

            static inline batch_type acos(const batch_type& z)
            {
                return acos_complex_impl(z);
            }

            static inline batch_type atan(const batch_type& z)
            {
                return atan_complex_impl(z);
            }
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, bool i3ec, std::size_t N>
        struct invtrigo_kernel<batch<xtl::xcomplex<T, T, i3ec>, N>>
        {
            using batch_type = batch<xtl::xcomplex<T, T, i3ec>, N>;

            static inline batch_type asin(const batch_type& z)
            {
                return asin_complex_impl(z);
            }

            static inline batch_type acos(const batch_type& z)
            {
                return acos_complex_impl(z);
            }

            static inline batch_type atan(const batch_type& z)
            {
                return atan_complex_impl(z);
            }
        };
#endif

        /********
         * sinh *
         ********/

        template <class T, std::size_t N>
        inline batch<T, N> sinh_complex_impl(const batch<T, N>& z)
        {
            auto x = z.real();
            auto y = z.imag();
            return batch<T, N>(sinh(x) * cos(y), cosh(x) * sin(y));
        }

        template <class T, std::size_t N>
        struct sinh_kernel<batch<std::complex<T>, N>>
        {
            using batch_type = batch<std::complex<T>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return sinh_complex_impl(z);
            }
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, bool i3ec, std::size_t N>
        struct sinh_kernel<batch<xtl::xcomplex<T, T, i3ec>, N>>
        {
            using batch_type = batch<xtl::xcomplex<T, T, i3ec>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return sinh_complex_impl(z);
            }
        };
#endif

        /********
         * cosh *
         ********/

        template <class T, std::size_t N>
        inline batch<T, N> cosh_complex_impl(const batch<T, N>& z)
        {
            auto x = z.real();
            auto y = z.imag();
            return batch<T, N>(cosh(x) * cos(y), sinh(x) * sin(y));
        }

        template <class T, std::size_t N>
        struct cosh_kernel<batch<std::complex<T>, N >>
        {
            using batch_type = batch<std::complex<T>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return cosh_complex_impl(z);
            }
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, bool i3ec, std::size_t N>
        struct cosh_kernel<batch<xtl::xcomplex<T, T, i3ec>, N>>
        {
            using batch_type = batch<xtl::xcomplex<T, T, i3ec>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return cosh_complex_impl(z);
            }
        };
#endif

        /********
         * tanh *
         ********/

        template <class T, std::size_t N>
        inline batch<T, N> tanh_complex_impl(const batch<T, N>& z)
        {
            using rvt = typename batch<T, N>::real_value_type;
            using real_batch = typename batch<T, N>::real_batch;
            auto x = z.real();
            auto y = z.imag();
            real_batch two = real_batch(rvt(2));
            auto d = cosh(two * x) + cos(two * y);
            return batch<T, N>(sinh(two * x) / d, sin(two * y) / d);
        }

        template <class T, std::size_t N>
        struct tanh_kernel<batch<std::complex<T>, N >>
        {
            using batch_type = batch<std::complex<T>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return tanh_complex_impl(z);
            }
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, bool i3ec, std::size_t N>
        struct tanh_kernel<batch<xtl::xcomplex<T, T, i3ec>, N>>
        {
            using batch_type = batch<xtl::xcomplex<T, T, i3ec>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return tanh_complex_impl(z);
            }
        };
#endif

        /*********
         * asinh *
         *********/

        template <class T, std::size_t N>
        inline batch<T, N> asinh_complex_impl(const batch<T, N>& z)
        {
            using b_type = batch<T, N>;
            b_type w = asin(b_type(-z.imag(), z.real()));
            w = b_type(w.imag(), -w.real());
            return w;
        }

        template <class T, std::size_t N>
        struct asinh_kernel<batch<std::complex<T>, N >>
        {
            using batch_type = batch<std::complex<T>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return asinh_complex_impl(z);
            }
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, bool i3ec, std::size_t N>
        struct asinh_kernel<batch<xtl::xcomplex<T, T, i3ec>, N>>
        {
            using batch_type = batch<xtl::xcomplex<T, T, i3ec>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return asinh_complex_impl(z);
            }
        };
#endif

        /*********
         * acosh *
         *********/

        template <class T, std::size_t N>
        inline batch<T, N> acosh_complex_impl(const batch<T, N>& z)
        {
            using b_type = batch<T, N>;
            b_type w = acos(z);
            w = b_type(-w.imag(), w.real());
            return w;
        }

        template <class T, std::size_t N>
        struct acosh_kernel<batch<std::complex<T>, N >>
        {
            using batch_type = batch<std::complex<T>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return acosh_complex_impl(z);
            }
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, bool i3ec, std::size_t N>
        struct acosh_kernel<batch<xtl::xcomplex<T, T, i3ec>, N>>
        {
            using batch_type = batch<xtl::xcomplex<T, T, i3ec>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return acosh_complex_impl(z);
            }
        };
#endif

        /*********
         * atanh *
         *********/

        template <class T, std::size_t N>
        inline batch<T, N> atanh_complex_impl(const batch<T, N>& z)
        {
            using b_type = batch<T, N>;
            b_type w = atan(b_type(-z.imag(), z.real()));
            w = b_type(w.imag(), -w.real());
            return w;
        }

        template <class T, std::size_t N>
        struct atanh_kernel<batch<std::complex<T>, N >>
        {
            using batch_type = batch<std::complex<T>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return atanh_complex_impl(z);
            }
        };

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        template <class T, bool i3ec, std::size_t N>
        struct atanh_kernel<batch<xtl::xcomplex<T, T, i3ec>, N>>
        {
            using batch_type = batch<xtl::xcomplex<T, T, i3ec>, N>;

            static inline batch_type compute(const batch_type& z)
            {
                return atanh_complex_impl(z);
            }
        };
#endif
    }

    template <class X>
    inline typename X::real_batch abs(const simd_complex_batch<X>& z)
    {
        return hypot(z.real(), z.imag());
    }

    template <class X>
    inline typename X::real_batch arg(const simd_complex_batch<X>& z)
    {
        return atan2(z.imag(), z.real());
    }

    template <class X>
    inline X conj(const simd_complex_batch<X>& z)
    {
        return X(z.real(), -z.imag());
    }

    namespace detail
    {
        template <class T>
        T csqrt_scale_factor() noexcept;

        template <class T>
        T csqrt_scale() noexcept;

        template <>
        inline float csqrt_scale_factor<float>() noexcept
        {
            return 6.7108864e7f;
        }

        template <>
        inline float csqrt_scale<float>() noexcept
        {
            return 1.220703125e-4f;
        }

        template <>
        inline double csqrt_scale_factor<double>() noexcept
        {
            return 1.8014398509481984e16;
        }

        template <>
        inline double csqrt_scale<double>() noexcept
        {
            return 7.450580596923828125e-9;
        }
    }

    template <class X>
    inline X sqrt(const simd_complex_batch<X>& z)
    {
        using real_batch = typename X::real_batch;
        using rvt = typename real_batch::value_type;
        real_batch x = z.real();
        real_batch y = z.imag();
        real_batch sqrt_x = sqrt(fabs(x));
        real_batch sqrt_hy = sqrt(0.5 * fabs(y));
        auto cond = (fabs(x) > real_batch(4.) || fabs(y) > real_batch(4.));
        x = select(cond, x * 0.25, x * detail::csqrt_scale_factor<rvt>());
        y = select(cond, y * 0.25, y * detail::csqrt_scale_factor<rvt>());
        real_batch scale = select(cond, real_batch(2.), real_batch(detail::csqrt_scale<rvt>()));
        real_batch r = abs(X(x, y));

        auto condxp = x > real_batch(0.);
        real_batch t0 = select(condxp, sqrt(0.5 * (r + x)), sqrt(0.5 * (r - x)));
        real_batch r0 = scale * fabs((0.5 * y) / t0);
        t0 *= scale;
        real_batch t = select(condxp, t0, r0);
        r = select(condxp, r0, t0);
        X resg = select(y < real_batch(0.), X(t, -r), X(t, r));
        real_batch ze(0.);

        return select(y == ze,
                      select(x == ze,
                             X(ze, ze),
                             select(x < ze, X(ze, sqrt_x), X(sqrt_x, ze))),
                      select(x == ze,
                             select(y > ze, X(sqrt_hy, sqrt_hy), X(sqrt_hy, -sqrt_hy)),
                             resg));
    }

    template <class X>
    inline typename simd_batch_traits<X>::real_batch
    norm(const simd_complex_batch<X>& rhs)
    {
        return rhs.real() * rhs.real() + rhs.imag() * rhs.imag();
    }

    template <class X>
    inline X proj(const simd_complex_batch<X>& rhs)
    {
        using real_batch = typename simd_batch_traits<X>::real_batch;
        auto cond = isinf(rhs.real()) || isinf(rhs.imag());
        return select(cond, X(infinity<real_batch>(), copysign(real_batch(0.), rhs.imag())), rhs);
    }
}

#endif
