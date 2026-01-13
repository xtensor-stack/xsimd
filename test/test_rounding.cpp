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
            for (size_t i = 0; i < nb_batches; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = ceil(in);
                detail::load_batch(ref, expected, i);
                INFO("ceil");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // floor
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::floor(v); });
            for (size_t i = 0; i < nb_batches; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = floor(in);
                detail::load_batch(ref, expected, i);
                INFO("floor");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // trunc
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::trunc(v); });
            for (size_t i = 0; i < nb_batches; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = trunc(in);
                detail::load_batch(ref, expected, i);
                INFO("trunc");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // round
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::round(v); });
            for (size_t i = 0; i < nb_batches; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = round(in);
                detail::load_batch(ref, expected, i);
                INFO("round");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // nearbyint
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::nearbyint(v); });
            for (size_t i = 0; i < nb_batches; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = nearbyint(in);
                detail::load_batch(ref, expected, i);
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
            for (size_t i = 0; i < nb_batches; i += size)
            {
                batch_type in;
                int_batch_type out, ref;
                detail::load_batch(in, input, i);
                out = nearbyint_as_int(in);
                detail::load_batch(ref, expected, i);
                INFO("nearbyint_as_int");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // rint
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::rint(v); });
            for (size_t i = 0; i < nb_batches; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = rint(in);
                detail::load_batch(ref, expected, i);
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
#endif
