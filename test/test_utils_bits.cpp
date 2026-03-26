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

#include <cstdint>

#include <doctest/doctest.h>

#include "xsimd/utils/bits.hpp"

TEST_CASE("[utils::make_bit_mask] single bit")
{
    CHECK_EQ(xsimd::utils::make_bit_mask<std::uint8_t>(0), 0x01);
    CHECK_EQ(xsimd::utils::make_bit_mask<std::uint8_t>(7), 0x80);
    CHECK_EQ(xsimd::utils::make_bit_mask<std::uint32_t>(0), 0x01u);
    CHECK_EQ(xsimd::utils::make_bit_mask<std::uint32_t>(31), 0x80000000u);
}

TEST_CASE("[utils::make_bit_mask] multiple bits")
{
    CHECK_EQ(xsimd::utils::make_bit_mask(0u, 1u), 0b11);
    CHECK_EQ(xsimd::utils::make_bit_mask<std::uint8_t>(0, 2, 4), 0b00010101);
}

TEST_CASE("[utils::all_bits_set] basic")
{
    CHECK(xsimd::utils::all_bits_set<0>(0x01u));
    CHECK(xsimd::utils::all_bits_set<7>(0x80u));
    CHECK_FALSE(xsimd::utils::all_bits_set<0>(0x00u));
    CHECK_FALSE(xsimd::utils::all_bits_set<1>(0x01u));
}

TEST_CASE("[utils::all_bits_set] multiple bits")
{
    CHECK((xsimd::utils::all_bits_set<0, 1>(0x03u)));
    CHECK((xsimd::utils::all_bits_set<0, 1>(0xFFu)));
    CHECK_FALSE((xsimd::utils::all_bits_set<0, 1>(0x01u)));
}

TEST_CASE("[utils::set_bit]")
{
    CHECK_EQ(xsimd::utils::set_bit<0>(0u), 0x01u);
    CHECK_EQ(xsimd::utils::set_bit<3>(0u), 0x08u);
    // Idempotent: setting an already-set bit
    CHECK_EQ(xsimd::utils::set_bit<0>(0x01u), 0x01u);
    // Does not clear other bits
    CHECK_EQ(xsimd::utils::set_bit<1>(0b1101u), 0b1111u);
}

TEST_CASE("[utils::make_low_mask]")
{
    CHECK_EQ(xsimd::utils::make_low_mask<std::uint8_t>(0), 0x00);
    CHECK_EQ(xsimd::utils::make_low_mask<std::uint8_t>(1), 0x01);
    CHECK_EQ(xsimd::utils::make_low_mask<std::uint8_t>(4), 0x0F);
    CHECK_EQ(xsimd::utils::make_low_mask<std::uint8_t>(7), 0x7F);
    // Full width
    CHECK_EQ(xsimd::utils::make_low_mask<std::uint8_t>(8), 0xFF);
    CHECK_EQ(xsimd::utils::make_low_mask<std::uint32_t>(32), 0xFFFFFFFFu);
    CHECK_EQ(xsimd::utils::make_low_mask<std::uint64_t>(64), 0xFFFFFFFFFFFFFFFFu);
}

enum class flag : std::uint32_t
{
    A = 0,
    B = 1,
    C = 4,
    D = 31,
};

TEST_CASE("[utils::uint_bitset] default construction")
{
    xsimd::utils::uint_bitset<flag, std::uint32_t> bs;
    CHECK_FALSE(bs.bit_is_set<flag::A>());
    CHECK_FALSE(bs.bit_is_set<flag::B>());
}

TEST_CASE("[utils::uint_bitset] construction from raw value")
{
    xsimd::utils::uint_bitset<flag, std::uint32_t> bs(0b11u);
    CHECK(bs.bit_is_set<flag::A>());
    CHECK(bs.bit_is_set<flag::B>());
    CHECK_FALSE(bs.bit_is_set<flag::C>());
}

TEST_CASE("[utils::uint_bitset] set_bit")
{
    xsimd::utils::uint_bitset<flag, std::uint32_t> bs;
    bs.set_bit<flag::A>();
    CHECK(bs.bit_is_set<flag::A>());
    CHECK_FALSE(bs.bit_is_set<flag::B>());
    bs.set_bit<flag::D>();
    CHECK(bs.bit_is_set<flag::D>());
}

TEST_CASE("[utils::uint_bitset] all_bits_set")
{
    xsimd::utils::uint_bitset<flag, std::uint32_t> bs(0b11u);
    CHECK((bs.all_bits_set<flag::A, flag::B>()));
    CHECK_FALSE((bs.all_bits_set<flag::A, flag::C>()));
}

TEST_CASE("[utils::uint_bitset] get_range")
{
    enum class rk : std::uint32_t
    {
        lo = 0,
        mid = 4,
        hi = 8
    };
    xsimd::utils::uint_bitset<rk, std::uint32_t> bs(0b10101011u);
    CHECK_EQ((bs.get_range<rk::lo, rk::mid>()), 0b1011u);
    CHECK_EQ((bs.get_range<rk::mid, rk::hi>()), 0b1010u);
}
