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

#include <cassert>
namespace xsimd
{
    namespace utils
    {
        template <typename I>
        constexpr I make_bit_mask(I bit)
        {
            assert(bit >= 0);
            assert(bit < static_cast<I>(8 * sizeof(I)));
            return static_cast<I>(I { 1 } << bit);
        }

        template <typename I, typename... Args>
        constexpr I make_bit_mask(I bit, Args... bits)
        {
            // TODO(C++17): Use fold expression
            return make_bit_mask<I>(bit) | make_bit_mask<I>(static_cast<I>(bits)...);
        }

        template <int... Bits, typename I>
        constexpr bool all_bits_set(I value)
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

        /* A bitset over an unsigned integer type, indexed by an enum key type. */
        template <typename K, typename U>
        struct uint_bitset
        {
            /* The underlying unsigned integer type storing the bits. */
            using storage_type = U;
            /* The enum type whose values name individual bits. */
            using key_type = K;

            /* Construct from a raw bit pattern. */
            constexpr explicit uint_bitset(storage_type bitset = {}) noexcept
                : m_bitset(bitset)
            {
            }

            /* Return true if every bit named by the template arguments is set. */
            template <key_type... bits>
            constexpr bool all_bits_set() const noexcept
            {
                return utils::all_bits_set<static_cast<storage_type>(bits)...>(m_bitset);
            }

            /* Return true if the bit is set. */
            template <key_type bit>
            constexpr bool bit_is_set() const noexcept
            {
                return all_bits_set<bit>();
            }

            /* Set the corresponding bit to true in the bitfield. */
            template <key_type bit>
            constexpr void set_bit() noexcept
            {
                m_bitset = utils::set_bit<static_cast<storage_type>(bit)>(m_bitset);
            }

        private:
            storage_type m_bitset = { 0 };
        };
    }
}

#endif
