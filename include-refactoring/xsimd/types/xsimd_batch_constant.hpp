/***************************************************************************
* Copyright (c) Serge Guelton                                              *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_BATCH_CONSTANT_HPP
#define XSIMD_BATCH_CONSTANT_HPP

#include "./xsimd_batch.hpp"

namespace xsimd
{
    template <class T, class A, bool... Values>
    struct batch_bool_constant
    {
        static constexpr std::size_t size = sizeof...(Values);
        using batch_type = batch_bool<T, A>;
        using value_type = typename batch_type::value_type;
        static_assert(sizeof...(Values) == batch_type::size, "consistent batch size");

        operator batch_type() const { return {Values...}; }

        bool get(size_t i) const
        {
            return std::array<value_type, size>{{Values...}}[i];
        }

        static constexpr int mask()
        {
            return mask_helper(0, static_cast<int>(Values)...);
        }

      private:
        static constexpr int mask_helper(int acc) { return acc; }
        template <class... Tys>
        static constexpr int mask_helper(int acc, int mask, Tys... masks)
        {
            return mask_helper(acc | mask, (masks << 1)...);
        }
    };

    template <class T, class A, T... Values>
    struct batch_constant
    {
        static constexpr std::size_t size = sizeof...(Values);
        using batch_type = batch<T, A>;
        using value_type = typename batch_type::value_type;
        static_assert(sizeof...(Values) == batch_type::size, "consistent batch size");

        operator batch_type() const { return {Values...}; }

        constexpr T get(size_t i) const
        {
            return std::array<value_type, size>{Values...}[i];
        }
    };

    namespace detail
    {
        template <class G, class A, std::size_t... Is>
        constexpr auto make_batch_constant(detail::index_sequence<Is...>)
            -> batch_constant<decltype(G::get(0, 0)), A,
                              G::get(Is, sizeof...(Is))...>
        {
            return {};
        }
        template <class T, class G, class A, std::size_t... Is>
        constexpr auto make_batch_bool_constant(detail::index_sequence<Is...>)
            -> batch_bool_constant<T, A, G::get(Is, sizeof...(Is))...>
        {
            return {};
        }

    } // namespace detail

    template <class G, class A=default_arch>
    constexpr auto make_batch_constant() -> decltype(
        detail::make_batch_constant<G, A>(detail::make_index_sequence<batch<decltype(G::get(0, 0)), A>::size>()))
    {
        return detail::make_batch_constant<G, A>(detail::make_index_sequence<batch<decltype(G::get(0, 0)), A>::size>());
    }

    template <class T, class G, class A=default_arch>
    constexpr auto make_batch_bool_constant()
        -> decltype(detail::make_batch_bool_constant<T, G, A>(
            detail::make_index_sequence<batch<T, A>::size>()))
    {
        return detail::make_batch_bool_constant<T, G, A>(
            detail::make_index_sequence<batch<T, A>::size>());
    }

} // namespace xsimd

#endif
