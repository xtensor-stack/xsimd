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
#include "./xsimd_utils.hpp"

namespace xsimd
{
    template <class batch_type, bool... Values>
    struct batch_bool_constant
    {
        static constexpr std::size_t size = sizeof...(Values);
        using arch_type = typename batch_type::arch_type;
        using value_type = bool;
        static_assert(sizeof...(Values) == batch_type::size, "consistent batch size");

        operator batch_bool<typename batch_type::value_type, arch_type>() const { return {Values...}; }

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

    template <class batch_type, typename batch_type::value_type... Values>
    struct batch_constant
    {
        static constexpr std::size_t size = sizeof...(Values);
        using arch_type = typename batch_type::arch_type;
        using value_type = typename batch_type::value_type;
        static_assert(sizeof...(Values) == batch_type::size, "consistent batch size");

        operator batch_type() const { return {Values...}; }

        constexpr value_type get(size_t i) const
        {
            return std::array<value_type, size>{Values...}[i];
        }
    };

    namespace detail
    {
        template <class batch_type, class G, std::size_t... Is>
        constexpr auto make_batch_constant(detail::index_sequence<Is...>)
            -> batch_constant<batch_type, G::get(Is, sizeof...(Is))...>
        {
            return {};
        }
        template <class batch_type, class G, std::size_t... Is>
        constexpr auto make_batch_bool_constant(detail::index_sequence<Is...>)
            -> batch_bool_constant<batch_type, G::get(Is, sizeof...(Is))...>
        {
            return {};
        }

    } // namespace detail

    template <class batch_type, class G>
    constexpr auto make_batch_constant() -> decltype(
        detail::make_batch_constant<batch_type, G>(detail::make_index_sequence<batch_type::size>()))
    {
        return detail::make_batch_constant<batch_type, G>(detail::make_index_sequence<batch_type::size>());
    }

    template <class batch_type, class G>
    constexpr auto make_batch_bool_constant()
        -> decltype(detail::make_batch_bool_constant<batch_type, G>(
            detail::make_index_sequence<batch_type::size>()))
    {
        return detail::make_batch_bool_constant<batch_type, G>(
            detail::make_index_sequence<batch_type::size>());
    }

} // namespace xsimd

#endif
