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
struct hyperbolic_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type input;
    vector_type acosh_input;
    vector_type atanh_input;
    vector_type expected;

    hyperbolic_test()
    {
        nb_input = size * 10000;
        input.resize(nb_input);
        acosh_input.resize(nb_input);
        atanh_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(-1.5) + i * value_type(3) / nb_input;
            acosh_input[i] = value_type(1.) + i * value_type(3) / nb_input;
            atanh_input[i] = value_type(-0.95) + i * value_type(1.9) / nb_input;
        }
        expected.resize(nb_input);
    }

    void test_hyperbolic_functions()
    {
        // sinh
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::sinh(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = sinh(in);
                detail::load_batch(ref, expected, i);
                INFO("sinh");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // cosh
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::cosh(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = cosh(in);
                detail::load_batch(ref, expected, i);
                INFO("cosh");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // tanh
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::tanh(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = tanh(in);
                detail::load_batch(ref, expected, i);
                INFO("tanh");
                CHECK_BATCH_EQ(ref, out);
            }
        }
    }

    void test_reciprocal_functions()
    {
        // asinh
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::asinh(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = asinh(in);
                detail::load_batch(ref, expected, i);
                INFO("asinh");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // acosh
        {
            std::transform(acosh_input.cbegin(), acosh_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::acosh(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, acosh_input, i);
                out = acosh(in);
                detail::load_batch(ref, expected, i);
                INFO("acosh");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // atanh
        {
            std::transform(atanh_input.cbegin(), atanh_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::atanh(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, atanh_input, i);
                out = atanh(in);
                detail::load_batch(ref, expected, i);
                INFO("atanh");
                CHECK_BATCH_EQ(ref, out);
            }
        }
    }
};

TEST_CASE_TEMPLATE("[hyperbolic]", B, BATCH_FLOAT_TYPES)
{
    hyperbolic_test<B> Test;

    SUBCASE("hyperbolic")
    {
        Test.test_hyperbolic_functions();
    }

    SUBCASE("reciprocal")
    {
        Test.test_reciprocal_functions();
    }
}
#endif
