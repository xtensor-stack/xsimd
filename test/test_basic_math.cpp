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

namespace detail
{
    template <class T, bool is_int = std::is_integral<typename T::value_type>::value>
    struct infinity_tester
    {
        static void test_isfinite()
        {
            T input(1);
            INFO("input: ", input);
            CHECK_UNARY(xsimd::all(xsimd::isfinite(input)));
        }

        static void test_isinf()
        {
            T input(1);
            INFO("input: ", input);
            CHECK_FALSE(xsimd::any(xsimd::isinf(input)));
        }
    };

    template <class T>
    struct infinity_tester<T, false>
    {
        static void test_isfinite()
        {
            T input = xsimd::infinity<T>();
            CHECK_FALSE(xsimd::any(xsimd::isfinite(input)));
        }
        static void test_isinf()
        {
            T input = xsimd::infinity<T>();
            CHECK_UNARY(xsimd::all(xsimd::isinf(input)));
        }
    };
}

template <class B>
struct basic_math_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;

    array_type lhs;
    array_type rhs;
    array_type clip_input;
    array_type from_input;

    basic_math_test()
    {
        for (size_t i = 0; i < size; ++i)
        {
            lhs[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25)) + value_type(1.);
            rhs[i] = value_type(10.2) / (i + 2) + value_type(0.25) + value_type(1.);
            clip_input[i] = i * value_type(0.25);
            from_input[i] = rhs[i] - value_type(1);
        }
    }

    void test_fmod() const
    {
        array_type expected;
        std::transform(
            lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
            [](const value_type& l, const value_type& r)
            { return std::fmod(l, r); });
        batch_type res = xsimd::fmod(batch_lhs(), batch_rhs());
        CHECK_BATCH_EQ(res, expected);
    }

    void test_remainder() const
    {
        array_type expected;
        std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                       [](const value_type& l, const value_type& r)
                       { return std::remainder(l, r); });
        batch_type res = xsimd::remainder(batch_lhs(), batch_rhs());
        CHECK_BATCH_EQ(res, expected);
    }

    void test_fdim() const
    {
        array_type expected;
        std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                       [](const value_type& l, const value_type& r)
                       { return std::fdim(l, r); });
        batch_type res = xsimd::fdim(batch_lhs(), batch_rhs());
        CHECK_BATCH_EQ(res, expected);
    }

    void test_clip()
    {
        value_type clip_lo = static_cast<value_type>(0.5);
        value_type clip_hi = static_cast<value_type>(1.);
        array_type expected;
        std::transform(clip_input.cbegin(), clip_input.cend(), expected.begin(),
                       [clip_lo, clip_hi](const value_type& l)
                       {
                           return l < clip_lo ? clip_lo : clip_hi < l ? clip_hi
                                                                      : l;
                       });
        batch_type res = xsimd::clip(batch_clip_input(), batch_type(clip_lo), batch_type(clip_hi));
        CHECK_BATCH_EQ(res, expected);
    }

    void test_isfinite()
    {
        detail::infinity_tester<batch_type>::test_isfinite();
    }

    void test_isinf()
    {
        detail::infinity_tester<batch_type>::test_isinf();
    }

    void test_nextafter()
    {
        array_type expected;
        std::transform(from_input.cbegin(), from_input.cend(), rhs.cbegin(), expected.begin(),
                       [](const value_type& l, const value_type& r)
                       { return std::nextafter(l, r); });
        batch_type res = xsimd::nextafter(batch_from_input(), batch_rhs());
        CHECK_BATCH_EQ(res, expected);
    }

private:
    batch_type batch_lhs() const
    {
        return batch_type::load_unaligned(lhs.data());
    }

    batch_type batch_rhs() const
    {
        return batch_type::load_unaligned(rhs.data());
    }

    batch_type batch_clip_input() const
    {
        return batch_type::load_unaligned(clip_input.data());
    }

    batch_type batch_from_input() const
    {
        return batch_type::load_unaligned(from_input.data());
    }
};

TEST_CASE_TEMPLATE("[basic math tests]", B, BATCH_MATH_TYPES)
{
    basic_math_test<B> Test;

    SUBCASE("fmod") { Test.test_fmod(); }
    SUBCASE("remainder") { Test.test_remainder(); }
    SUBCASE("fdim") { Test.test_fdim(); }
    SUBCASE("clip") { Test.test_clip(); }
    SUBCASE("isfinite") { Test.test_isfinite(); }
    SUBCASE("isinf") { Test.test_isinf(); }
    SUBCASE("nextafter") { Test.test_nextafter(); }
}
#endif
