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

template <class B>
struct power_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type zero_input;
    vector_type zlhs_input;
    vector_type lhs_input;
    vector_type rhs_input;
    vector_type expected;

    power_test()
    {
        nb_input = size * 10000;
        zero_input.resize(nb_input);
        zlhs_input.resize(nb_input);
        lhs_input.resize(nb_input);
        rhs_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            zero_input[i] = 0;
            lhs_input[i] = value_type(i / 4 + 1.2 * std::sqrt(i + 0.25));
            zlhs_input[i] = lhs_input[i] * (i % 2);
            rhs_input[i] = value_type(10.2 / (i + 2) + 0.25);
        }

        expected.resize(nb_input);
    }

    void test_power_functions()
    {
        // pow
        {
            std::transform(lhs_input.cbegin(), lhs_input.cend(), rhs_input.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return std::pow(l, r); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type lhs_in, rhs_in, out, ref;
                detail::load_batch(lhs_in, lhs_input, i);
                detail::load_batch(rhs_in, rhs_input, i);
                out = pow(lhs_in, rhs_in);
                detail::load_batch(ref, expected, i);
                INFO("pow");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // pow zero
        {
            std::transform(zlhs_input.cbegin(), zlhs_input.cend(), rhs_input.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return std::pow(l, r); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type zlhs_in, rhs_in, out, ref;
                detail::load_batch(zlhs_in, zlhs_input, i);
                detail::load_batch(rhs_in, rhs_input, i);
                out = pow(zlhs_in, rhs_in);
                detail::load_batch(ref, expected, i);
                INFO("0 ^ x");
                CHECK_BATCH_EQ(ref, out);
            }

// use of undeclared identifier '_MM_SET_EXCEPTION_MASK for emscripten
#if defined(__SSE__) && !defined(EMSCRIPTEN)
            // Test with FE_INVALID...
            unsigned mask = _MM_GET_EXCEPTION_MASK();
            _MM_SET_EXCEPTION_MASK(mask & ~_MM_MASK_INVALID);
            std::vector<value_type> computed(nb_input);
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type zlhs_in, rhs_in, out;
                detail::load_batch(zlhs_in, zlhs_input, i);
                detail::load_batch(rhs_in, rhs_input, i);
                out = pow(zlhs_in, rhs_in);
                detail::store_batch(out, computed, i);
            }
            _MM_SET_EXCEPTION_MASK(mask);
            // Compare after the restore: the comparison raises FE_INVALID on
            // inf/NaN lanes and would trap with the exception unmasked.
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type ref, out;
                detail::load_batch(ref, expected, i);
                detail::load_batch(out, computed, i);
                INFO("0 ^ x with exception");
                CHECK_BATCH_EQ(ref, out);
            }
#endif
        }
#ifndef __FAST_MATH__
        // pow 0^-x
        {
            std::transform(zero_input.cbegin(), zero_input.cend(), rhs_input.cbegin(), expected.begin(),
                           [](const value_type& z, const value_type& r)
                           { return std::pow(z, -r); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type zero_in, rhs_in, out, ref;
                detail::load_batch(zero_in, zero_input, i);
                detail::load_batch(rhs_in, rhs_input, i);
                out = pow(zero_in, -rhs_in);
                detail::load_batch(ref, expected, i);
                INFO("pow(0, -x)");
                CHECK_BATCH_EQ(ref, out);
            }
        }
#endif
        // ipow
        {
            long k = 0;
            std::transform(lhs_input.cbegin(), lhs_input.cend(), expected.begin(),
                           [&k, this](const value_type& l)
                           { auto arg = k / size / 8000 - nb_input / size / 8000 / 2; ++k; return std::pow(l, arg); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type lhs_in, out, ref;
                detail::load_batch(lhs_in, lhs_input, i);
                out = pow(lhs_in, i / size / 8000 - nb_input / size / 8000 / 2);
                detail::load_batch(ref, expected, i);
                INFO("ipow");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // hypot
        {
            std::transform(lhs_input.cbegin(), lhs_input.cend(), rhs_input.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return std::hypot(l, r); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type lhs_in, rhs_in, out, ref;
                detail::load_batch(lhs_in, lhs_input, i);
                detail::load_batch(rhs_in, rhs_input, i);
                out = hypot(lhs_in, rhs_in);
                detail::load_batch(ref, expected, i);
                INFO("hypot");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // cbrt
        {
            std::transform(lhs_input.cbegin(), lhs_input.cend(), expected.begin(),
                           [](const value_type& l)
                           { return std::cbrt(l); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type lhs_in, out, ref;
                detail::load_batch(lhs_in, lhs_input, i);
                out = cbrt(lhs_in);
                detail::load_batch(ref, expected, i);
                INFO("cbrt");
                CHECK_BATCH_EQ(ref, out);
            }
        }
    }
};

TEST_CASE_TEMPLATE("[power]", B, BATCH_FLOAT_TYPES)
{
    power_test<B> Test;
    Test.test_power_functions();
}

TEST_CASE_TEMPLATE("[power] special cases", B, BATCH_FLOAT_TYPES)
{
    using value_type = typename B::value_type;
    struct test_case
    {
        value_type base;
        value_type exponent;
        value_type expected;
    };
#ifndef __FAST_MATH__
    const auto nan = std::numeric_limits<value_type>::quiet_NaN();
    const auto inf = std::numeric_limits<value_type>::infinity();
#endif
    const test_case cases[] = {
        { 0, 0, 1 },
        { 0, 2, 0 },
        { 0, -0., 1 },
        { -1, 3, -1 },
        { -0., 0, 1 },
        { 2, 3, 8 },
        { 2, 0, 1 },
        { 1, 2, 1 },
#ifndef __FAST_MATH__
        { nan, 0, 1 },
        { inf, 0, 1 },
        { -inf, -0., 1 },
        { 1, nan, 1 },
        { 1, inf, 1 },
        { 1, -inf, 1 },
        { -1, inf, 1 },
        { -1, -inf, 1 },
        { -1, nan, nan },
        { -1, 0.5, nan },
#endif
    };
    constexpr size_t case_count = sizeof(cases) / sizeof(cases[0]);
    for (size_t i = 0; i < case_count; i += B::size)
    {
        std::array<value_type, B::size> bases, exponents, expected;
        for (size_t lane = 0; lane < B::size; ++lane)
        {
            const auto& c = cases[(i + lane) % case_count];
            bases[lane] = c.base;
            exponents[lane] = c.exponent;
            expected[lane] = c.expected;
        }
        const auto actual = xsimd::pow(B::load_unaligned(bases.data()), B::load_unaligned(exponents.data()));
        for (size_t lane = 0; lane < B::size; ++lane)
        {
            const auto result = actual.get(lane);
            CAPTURE(bases[lane]);
            CAPTURE(exponents[lane]);
            CAPTURE(expected[lane]);
            CAPTURE(result);
            CHECK(detail::expect_scalar_near(expected[lane], result));
        }
    }
}
#endif
