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
struct complex_hyperbolic_test
{
    using batch_type = B;
    using real_batch_type = typename B::real_batch;
    using value_type = typename B::value_type;
    using real_value_type = typename value_type::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type input;
    vector_type acosh_input;
    vector_type atanh_input;
    vector_type expected;

    complex_hyperbolic_test()
    {
        nb_input = 10000 * size;
        input.resize(nb_input);
        acosh_input.resize(nb_input);
        atanh_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(real_value_type(-1.5) + i * real_value_type(3) / nb_input,
                                  real_value_type(-1.3) + i * real_value_type(2.5) / nb_input);
            acosh_input[i] = value_type(real_value_type(1.) + i * real_value_type(3) / nb_input,
                                        real_value_type(1.2) + i * real_value_type(2.7) / nb_input);
            atanh_input[i] = value_type(real_value_type(-0.95) + i * real_value_type(1.9) / nb_input,
                                        real_value_type(-0.94) + i * real_value_type(1.8) / nb_input);
        }
        expected.resize(nb_input);
    }

    void test_sinh()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::sinh; return sinh(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, input, i);
            out = sinh(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }

    void test_cosh()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::cosh; return cosh(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, input, i);
            out = cosh(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }

    void test_tanh()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::tanh; return tanh(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, input, i);
            out = tanh(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }

    void test_asinh()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::asinh; return asinh(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, input, i);
            out = asinh(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }

    void test_acosh()
    {
        std::transform(acosh_input.cbegin(), acosh_input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::acosh; return acosh(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, acosh_input, i);
            out = acosh(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }

    void test_atanh()
    {
        std::transform(atanh_input.cbegin(), atanh_input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::atanh; return atanh(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, atanh_input, i);
            out = atanh(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }
};

TEST_CASE_TEMPLATE("[complex hyperbolic]", B, BATCH_COMPLEX_TYPES)
{
    complex_hyperbolic_test<B> Test;
    SUBCASE("sinh")
    {
        Test.test_sinh();
    }

    SUBCASE("cosh")
    {
        Test.test_cosh();
    }

    SUBCASE("tanh")
    {
        Test.test_tanh();
    }

    SUBCASE("asinh")
    {
        Test.test_asinh();
    }

    SUBCASE("acosh")
    {
        Test.test_acosh();
    }

    SUBCASE("atanh")
    {
        Test.test_atanh();
    }
}
#endif
