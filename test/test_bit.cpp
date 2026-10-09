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

#include "xsimd/xsimd.hpp"
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "test_utils.hpp"

// std::popcount is constexpr and so are __builtin_popcount and
// __builtin_popcountll on GCC and Clang, so the backport popcount is a
// constant expression on the MSVC and fallback paths too wherever a builtin
// is available.
#if XSIMD_CPP_VERSION >= 202002L
#include <version>
#endif
#if (XSIMD_CPP_VERSION >= 202002L && __cpp_lib_bitops >= 201907L) || defined(__GNUC__) || defined(__clang__)
static_assert(xsimd::detail::popcount(0xffu) == 8, "popcount must be usable in a constant expression");
#endif

template <class T>
struct bit_test
{
    using value_type = T;
    using bits = std::integral_constant<int, sizeof(T) * CHAR_BIT>;

    void test_popcount()
    {
        auto check = [&](T v, int ref)
        {
            INFO("popcount(0x" << std::hex << (unsigned long long)v << std::dec << ")");
            CHECK_EQ(xsimd::detail::popcount(v), ref);
#if !(XSIMD_CPP_VERSION >= 202002L && __cpp_lib_bitops >= 201907L)
            // Run the portable fallback directly: on a compiler with a
            // builtin the dispatch never reaches it.
            CHECK_EQ(xsimd::detail::popcount_swar(v), ref);
#endif
        };

        check(T(0), 0);
        check(T(~T(0)), bits::value);
        for (int i = 0; i < bits::value; ++i)
        {
            check(T(T(1) << i), 1);
            check(T(T(~T(0)) >> i), bits::value - i);
        }

        // Alternating patterns
        if (bits::value >= 8)
        {
            T pattern_aa = T(0);
            T pattern_55 = T(0);
            for (int i = 0; i < bits::value / 8; ++i)
            {
                pattern_aa |= T(0xAA) << (i * 8);
                pattern_55 |= T(0x55) << (i * 8);
            }
            check(pattern_aa, bits::value / 2);
            check(pattern_55, bits::value / 2);
        }

        // Specific test cases
        check(T(1), 1);
        check(T(3), 2);
        check(T(7), 3);
        check(T(15), 4);
    }

    void test_countl_zero()
    {
        // Zero should have all leading zeros
        CHECK_EQ(xsimd::detail::countl_zero(T(0)), bits::value);

        // All bits set should have 0 leading zeros
        CHECK_EQ(xsimd::detail::countl_zero(T(~T(0))), 0);

        // MSB set should have 0 leading zeros
        T msb = T(1) << (bits::value - 1);
        CHECK_EQ(xsimd::detail::countl_zero(msb), 0);

        // Powers of 2
        for (int i = 0; i < bits::value; ++i)
        {
            T value = T(T(1) << i);
            int expected = bits::value - i - 1;
            INFO("countl_zero(1 << " << i << ")");
            CHECK_EQ(xsimd::detail::countl_zero(value), expected);
        }

        // Sequential patterns (1, 3, 7, 15, ...)
        for (int i = 1; i < bits::value; ++i)
        {
            T value = T((T(1) << i) - 1);
            int expected = bits::value - i;
            INFO("countl_zero((1 << " << i << ") - 1)");
            CHECK_EQ(xsimd::detail::countl_zero(value), expected);
        }

        // Specific values
        CHECK_EQ(xsimd::detail::countl_zero(T(1)), bits::value - 1);
        CHECK_EQ(xsimd::detail::countl_zero(T(2)), bits::value - 2);
        CHECK_EQ(xsimd::detail::countl_zero(T(4)), bits::value - 3);
    }

    void test_countl_one()
    {
        // Zero should have 0 leading ones
        CHECK_EQ(xsimd::detail::countl_one(T(0)), 0);

        // All bits set should have all leading ones
        CHECK_EQ(xsimd::detail::countl_one(T(~T(0))), bits::value);

        // MSB clear, rest set should have 0 leading ones
        T pattern = T(~(T(1) << (bits::value - 1)));
        CHECK_EQ(xsimd::detail::countl_one(pattern), 0);

        // Inverted powers of 2
        for (int i = 0; i < bits::value; ++i)
        {
            T value = T(~(T(1) << i));
            int expected = (i == bits::value - 1) ? 0 : bits::value - i - 1;
            INFO("countl_one(~(1 << " << i << "))");
            CHECK_EQ(xsimd::detail::countl_one(value), expected);
        }

        // Patterns with known leading ones
        for (int i = 1; i <= bits::value; ++i)
        {
            T value = T(T(~T(0)) << (bits::value - i));
            INFO("countl_one(~0 << " << (bits::value - i) << ")");
            CHECK_EQ(xsimd::detail::countl_one(value), i);
        }

        // Specific values
        CHECK_EQ(xsimd::detail::countl_one(T(~T(1))), bits::value - 1);
        CHECK_EQ(xsimd::detail::countl_one(T(~T(3))), bits::value - 2);
    }

    void test_countr_zero()
    {
        // Zero should have all trailing zeros
        CHECK_EQ(xsimd::detail::countr_zero(T(0)), bits::value);

        // All bits set should have 0 trailing zeros
        CHECK_EQ(xsimd::detail::countr_zero(T(~T(0))), 0);

        // Odd numbers should have 0 trailing zeros
        CHECK_EQ(xsimd::detail::countr_zero(T(1)), 0);
        CHECK_EQ(xsimd::detail::countr_zero(T(3)), 0);
        CHECK_EQ(xsimd::detail::countr_zero(T(5)), 0);
        CHECK_EQ(xsimd::detail::countr_zero(T(7)), 0);

        // Powers of 2
        for (int i = 0; i < bits::value; ++i)
        {
            T value = T(1) << i;
            INFO("countr_zero(1 << " << i << ")");
            CHECK_EQ(xsimd::detail::countr_zero(value), i);
        }

        // Even numbers with known factors
        CHECK_EQ(xsimd::detail::countr_zero(T(2)), 1);
        CHECK_EQ(xsimd::detail::countr_zero(T(4)), 2);
        CHECK_EQ(xsimd::detail::countr_zero(T(6)), 1);
        CHECK_EQ(xsimd::detail::countr_zero(T(8)), 3);
        CHECK_EQ(xsimd::detail::countr_zero(T(12)), 2);
        CHECK_EQ(xsimd::detail::countr_zero(T(16)), 4);

        // Specific patterns
        for (int i = 1; i < bits::value; ++i)
        {
            T value = T(~T(0)) << i;
            INFO("countr_zero(~0 << " << i << ")");
            CHECK_EQ(xsimd::detail::countr_zero(value), i);
        }
    }

    void test_countr_one()
    {
        // Zero should have 0 trailing ones
        CHECK_EQ(xsimd::detail::countr_one(T(0)), 0);

        // All bits set should have all trailing ones
        CHECK_EQ(xsimd::detail::countr_one(T(~T(0))), bits::value);

        // Even numbers should have 0 trailing ones
        CHECK_EQ(xsimd::detail::countr_one(T(2)), 0);
        CHECK_EQ(xsimd::detail::countr_one(T(4)), 0);
        CHECK_EQ(xsimd::detail::countr_one(T(6)), 0);

        // Powers of 2 minus 1
        for (int i = 1; i < bits::value; ++i)
        {
            T value = T((T(1) << i) - 1);
            INFO("countr_one((1 << " << i << ") - 1)");
            CHECK_EQ(xsimd::detail::countr_one(value), i);
        }

        // Specific values
        CHECK_EQ(xsimd::detail::countr_one(T(1)), 1);
        CHECK_EQ(xsimd::detail::countr_one(T(3)), 2);
        CHECK_EQ(xsimd::detail::countr_one(T(7)), 3);
        CHECK_EQ(xsimd::detail::countr_one(T(15)), 4);
        CHECK_EQ(xsimd::detail::countr_one(T(31)), 5);

        // Inverted powers of 2 minus 1
        for (int i = 1; i < bits::value; ++i)
        {
            T value = T(~((T(1) << i) - 1));
            INFO("countr_one(~((1 << " << i << ") - 1))");
            CHECK_EQ(xsimd::detail::countr_one(value), 0);
        }
    }
};

TEST_CASE_TEMPLATE("[bit operations]", T,
                   uint8_t, uint16_t, uint32_t, uint64_t)
{
    bit_test<T> Test;

    SUBCASE("popcount") { Test.test_popcount(); }
    SUBCASE("countl_zero") { Test.test_countl_zero(); }
    SUBCASE("countl_one") { Test.test_countl_one(); }
    SUBCASE("countr_zero") { Test.test_countr_zero(); }
    SUBCASE("countr_one") { Test.test_countr_one(); }
}

#endif
