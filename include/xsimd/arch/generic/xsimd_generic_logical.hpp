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

#ifndef XSIMD_GENERIC_LOGICAL_HPP
#define XSIMD_GENERIC_LOGICAL_HPP

#include "./xsimd_generic_details.hpp"

namespace xsimd
{

    namespace kernel
    {

        using namespace types;

        // ge
        template <class A, class T>
        inline batch_bool<T, A> ge(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) noexcept
        {
            return other <= self;
        }

        // gt
        template <class A, class T>
        inline batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) noexcept
        {
            return other < self;
        }

        // is_even
        template <class A, class T>
        inline batch_bool<T, A> is_even(batch<T, A> const& self, requires_arch<generic>) noexcept
        {
            return is_flint(self * T(0.5));
        }

        // is_flint
        template <class A, class T>
        inline batch_bool<T, A> is_flint(batch<T, A> const& self, requires_arch<generic>) noexcept
        {
            auto frac = select(isnan(self - self), constants::nan<batch<T, A>>(), self - trunc(self));
            return frac == T(0.);
        }

        // is_odd
        template <class A, class T>
        inline batch_bool<T, A> is_odd(batch<T, A> const& self, requires_arch<generic>) noexcept
        {
            return is_even(self - T(1.));
        }

        // isinf
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> isinf(batch<T, A> const&, requires_arch<generic>) noexcept
        {
            return batch_bool<T, A>(false);
        }
        template <class A>
        inline batch_bool<float, A> isinf(batch<float, A> const& self, requires_arch<generic>) noexcept
        {
            return abs(self) == std::numeric_limits<float>::infinity();
        }
        template <class A>
        inline batch_bool<double, A> isinf(batch<double, A> const& self, requires_arch<generic>) noexcept
        {
            return abs(self) == std::numeric_limits<double>::infinity();
        }

        // isfinite
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> isfinite(batch<T, A> const&, requires_arch<generic>) noexcept
        {
            return batch_bool<T, A>(true);
        }
        template <class A>
        inline batch_bool<float, A> isfinite(batch<float, A> const& self, requires_arch<generic>) noexcept
        {
            return (self - self) == 0.f;
        }
        template <class A>
        inline batch_bool<double, A> isfinite(batch<double, A> const& self, requires_arch<generic>) noexcept
        {
            return (self - self) == 0.;
        }

        // isnan
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> isnan(batch<T, A> const&, requires_arch<generic>) noexcept
        {
            return batch_bool<T, A>(false);
        }

        // le
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> le(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) noexcept
        {
            return (self < other) || (self == other);
        }

        // neq
        template <class A, class T>
        inline batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) noexcept
        {
            return !(other == self);
        }

        // logical_and
        template <class A, class T>
        inline batch<T, A> logical_and(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) noexcept
        {
            return detail::apply([](T x, T y) noexcept
                                 { return x && y; },
                                 self, other);
        }

        // logical_or
        template <class A, class T>
        inline batch<T, A> logical_or(batch<T, A> const& self, batch<T, A> const& other, requires_arch<generic>) noexcept
        {
            return detail::apply([](T x, T y) noexcept
                                 { return x || y; },
                                 self, other);
        }
    }
}

#endif
