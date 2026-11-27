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

    // Keep the inlined pow kernel out of the fixture's stack frame on MSVC.
    static DOCTEST_NOINLINE batch_type pow_batch(const batch_type& base, const batch_type& exponent)
    {
        return pow(base, exponent);
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
                out = pow_batch(lhs_in, rhs_in);
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
                out = pow_batch(zlhs_in, rhs_in);
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
                out = pow_batch(zlhs_in, rhs_in);
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
        CHECK(xsimd::all(pow_batch(batch_type(0), batch_type(0)) == batch_type(1)));
        // Squares and cubes avoid the logarithm/exponential rounding.
        for (const auto& c : { std::array<value_type, 3> { 3, 2, 9 }, { 4, 2, 16 }, { 5, 2, 25 }, { 2, 3, 8 }, { 7, 3, 343 }, { 10, 2, 100 } })
        {
            CHECK(xsimd::all(pow_batch(batch_type(c[0]), batch_type(c[1])) == batch_type(c[2])));
        }
        const auto mixed_base = xsimd::insert(batch_type(16), value_type(-7), xsimd::index<0>());
        const auto mixed_exponent = xsimd::insert(batch_type(0.5), value_type(3), xsimd::index<0>());
        const auto mixed_result = pow_batch(mixed_base, mixed_exponent);
        CHECK(mixed_result.get(0) == value_type(-343));
        CHECK_BATCH_EQ(xsimd::insert(batch_type(4), value_type(-343), xsimd::index<0>()), mixed_result);
#ifndef __FAST_MATH__
        const auto nan = std::numeric_limits<value_type>::quiet_NaN();
        const auto inf = std::numeric_limits<value_type>::infinity();
        CHECK(std::signbit(pow_batch(batch_type(-value_type(0)), batch_type(3)).get(0)));
        CHECK(xsimd::all(pow_batch(batch_type(-inf), batch_type(2)) == batch_type(inf)));
        CHECK(xsimd::all(xsimd::isnan(pow_batch(batch_type(nan), batch_type(3)))));
        CHECK(std::isnan(pow_batch(xsimd::insert(batch_type(16), nan, xsimd::index<0>()), mixed_exponent).get(0)));
        CHECK(xsimd::all(pow_batch(batch_type(nan), batch_type(0)) == batch_type(1)));
        CHECK(xsimd::all(pow_batch(batch_type(1), batch_type(nan)) == batch_type(1)));
        CHECK(xsimd::all(pow_batch(batch_type(-1), batch_type(inf)) == batch_type(1)));
        // Subtracting one here rounds to an even integer.
        const auto large_even = std::ldexp(value_type(1), std::numeric_limits<value_type>::digits) + value_type(2);
        CHECK(xsimd::all(pow_batch(batch_type(-1), batch_type(large_even)) == batch_type(1)));

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
                out = pow_batch(zero_in, -rhs_in);
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
#endif
