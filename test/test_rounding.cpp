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

#include <cfenv>

template <class B>
struct rounding_test
{
    using batch_type = B;
    using arch_type = typename B::arch_type;
    using value_type = typename B::value_type;
    using int_value_type = xsimd::as_integer_t<value_type>;
    using int_batch_type = xsimd::batch<int_value_type, arch_type>;
    static constexpr size_t size = B::size;
    static constexpr size_t nb_input = 16;
    static constexpr size_t nb_batches = nb_input / size;

    std::array<value_type, nb_input> input;
    std::array<value_type, nb_input> expected;

    rounding_test()
    {
        input[0] = value_type(-3.7);
        input[1] = value_type(-3.5);
        input[2] = value_type(-3.3);
        input[3] = value_type(-3.1);
        input[4] = value_type(-2.9);
        input[5] = value_type(-2.0);
        input[6] = value_type(-1.9);
        input[7] = value_type(-0.5);
        input[8] = value_type(0.5);
        input[9] = value_type(1.9);
        input[10] = value_type(2.0);
        input[11] = value_type(2.9);
        input[12] = value_type(3.1);
        input[13] = value_type(3.3);
        input[14] = value_type(3.5);
        input[15] = value_type(3.7);
    }

    void test_rounding_functions()
    {
        // ceil
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::ceil(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i * size);
                out = ceil(in);
                detail::load_batch(ref, expected, i * size);
                INFO("ceil");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // floor
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::floor(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i * size);
                out = floor(in);
                detail::load_batch(ref, expected, i * size);
                INFO("floor");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // trunc
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::trunc(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i * size);
                out = trunc(in);
                detail::load_batch(ref, expected, i * size);
                INFO("trunc");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // round
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::round(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i * size);
                out = round(in);
                detail::load_batch(ref, expected, i * size);
                INFO("round");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // nearbyint
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::nearbyint(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i * size);
                out = nearbyint(in);
                detail::load_batch(ref, expected, i * size);
                INFO("nearbyint");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // nearbyint_as_int
        {
            std::array<int_value_type, nb_input> expected;
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return xsimd::nearbyint_as_int(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in;
                int_batch_type out, ref;
                detail::load_batch(in, input, i * size);
                out = nearbyint_as_int(in);
                detail::load_batch(ref, expected, i * size);
                INFO("nearbyint_as_int");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // rint
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::rint(v); });
            for (size_t i = 0; i < nb_batches; i++)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i * size);
                out = rint(in);
                detail::load_batch(ref, expected, i * size);
                INFO("rint");
                CHECK_BATCH_EQ(ref, out);
            }
        }
    }
};

TEST_CASE_TEMPLATE("[rounding]", B, BATCH_FLOAT_TYPES)
{

    rounding_test<B> Test;
    Test.test_rounding_functions();
}

#ifndef __FAST_MATH__
TEST_CASE_TEMPLATE("[rounding] current rounding mode", B, BATCH_FLOAT_TYPES)
{
    using value_type = typename B::value_type;
    struct rounding_mode_guard
    {
        int saved = std::fegetround();
        ~rounding_mode_guard() { std::fesetround(saved); }
    } guard;

    const std::array<value_type, 16> input = {
        0., -0., 0.25, -0.25, 0.5, -0.5, 1.25, -1.25,
        1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 2., -2.
    };
    const std::array<int, 4> modes = { FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO };
    const value_type expected[][16] = {
        { 0., -0., 0., -0., 0., -0., 1., -1., 2., -2., 2., -2., 4., -4., 2., -2. },
        { 0., -0., 0., -1., 0., -1., 1., -2., 1., -2., 2., -3., 3., -4., 2., -2. },
        { 0., -0., 1., -0., 1., -0., 2., -1., 2., -1., 3., -2., 4., -3., 2., -2. },
        { 0., -0., 0., -0., 0., -0., 1., -1., 1., -1., 2., -2., 3., -3., 2., -2. }
    };

    for (size_t mode = 0; mode < modes.size(); ++mode)
    {
        if (std::fesetround(modes[mode]) != 0)
        {
            continue;
        }
        REQUIRE(std::fegetround() == modes[mode]);
        for (size_t offset = 0; offset < input.size(); offset += B::size)
        {
            std::array<value_type, B::size> values, nearby, rounded;
            for (size_t lane = 0; lane < B::size; ++lane)
            {
                // Reload after changing the mode so arithmetic cannot be hoisted.
                volatile value_type value = input[(offset + lane) % input.size()];
                values[lane] = value;
            }
            B batch = B::load_unaligned(values.data());
            xsimd::nearbyint(batch).store_unaligned(nearby.data());
            xsimd::rint(batch).store_unaligned(rounded.data());
            for (size_t lane = 0; lane < B::size; ++lane)
            {
                CAPTURE(modes[mode]);
                CAPTURE(values[lane]);
                value_type reference = expected[mode][(offset + lane) % input.size()];
                CHECK(nearby[lane] == reference);
                CHECK(rounded[lane] == reference);
                CHECK(std::signbit(nearby[lane]) == std::signbit(reference));
                CHECK(std::signbit(rounded[lane]) == std::signbit(reference));
            }
        }
    }
}
#endif
#endif
