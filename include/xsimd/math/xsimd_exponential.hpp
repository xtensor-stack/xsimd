/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_EXPONENTIAL_HPP
#define XSIMD_EXPONENTIAL_HPP

#include "xsimd_exp_reduction.hpp"
#include "xsimd_fp_manipulation.hpp"

namespace xsimd
{
    template <class T, std::size_t N>
    batch<T, N> exp(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch<T, N> exp2(const batch<T, N>& x);

    template <class T, std::size_t N>
    batch<T, N> exp10(const batch<T, N>& x);

    /******************************
     * exponential implementation *
     ******************************/

    namespace detail
    {
        template <class B, class Tag, class T = typename B::value_type>
        struct exponential;

        template <class B, class Tag>
        struct exponential<B, Tag, float>
        {
            static inline B compute(const B& a)
            {
                using reducer_t = exp_reduction<B, Tag>;
                B x;
                B k = reducer_t::reduce(a, x);
                x = reducer_t::approx(x);
                x = select(a <= reducer_t::minlog(), B(0.), ldexp(x, to_int(k)));
                x = select(a >= reducer_t::maxlog(), infinity<B>(), x);
                return x;
            }
        };

        template <class B, class Tag>
        struct exponential<B, Tag, double>
        {
            static inline B compute(const B& a)
            {
                using reducer_t = exp_reduction<B, Tag>;
                B hi, lo, x;
                B k = reducer_t::reduce(a, hi, lo, x);
                B c = reducer_t::approx(x);
                c = reducer_t::finalize(x, c, hi, lo);
                c = select(a <= reducer_t::minlog(), B(0.), ldexp(c, to_int(k)));
                c = select(a >= reducer_t::maxlog(), infinity<B>(), c);
                return c;
            }
        };
    }

    template <class T, std::size_t N>
    inline batch<T, N> exp(const batch<T, N>& x)
    {
        return detail::exponential<batch<T, N>, exp_tag>::compute(x);
    }

    template <class T, std::size_t N>
    inline batch<T, N> exp2(const batch<T, N>& x)
    {
        return detail::exponential<batch<T, N>, exp2_tag>::compute(x);
    }

    template <class T, std::size_t N>
    inline batch<T, N> exp10(const batch<T, N>& x)
    {
        return detail::exponential<batch<T, N>, exp10_tag>::compute(x);
    }

}

#endif
