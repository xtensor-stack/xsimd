/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ***************************************************************************/

#ifndef XSIMD_CPUID_UTILS_HPP
#define XSIMD_CPUID_UTILS_HPP

namespace xsimd
{
    namespace utils
    {
        template <typename I>
        constexpr I make_bit_mask(I bit)
        {
            return static_cast<I>(I { 1 } << bit);
        }

        template <typename I, typename... Args>
        constexpr I make_bit_mask(I bit, Args... bits)
        {
            // TODO(C++17): Use fold expression
            return make_bit_mask<I>(bit) | make_bit_mask<I>(static_cast<I>(bits)...);
        }

        template <int... Bits, typename I>
        constexpr bool bit_is_set(I value)
        {
            constexpr I mask = make_bit_mask<I>(static_cast<I>(Bits)...);
            return (value & mask) == mask;
        }

        template <int Bit, typename I>
        constexpr I set_bit(I value)
        {
            constexpr I mask = make_bit_mask<I>(static_cast<I>(Bit));
            return value | mask;
        }
    }
}

#endif
