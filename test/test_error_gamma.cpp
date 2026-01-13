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
struct error_gamma_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type input;
    vector_type gamma_input;
    vector_type gamma_neg_input;
    vector_type expected;

    error_gamma_test()
    {
        nb_input = size * 10000;
        input.resize(nb_input);
        gamma_input.resize(nb_input);
        gamma_neg_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(-1.5) + i * value_type(3) / nb_input;
            gamma_input[i] = value_type(0.5) + i * value_type(3) / nb_input;
            gamma_neg_input[i] = value_type(-3.99) + i * value_type(0.9) / nb_input;
        }
        expected.resize(nb_input);
    }

    void test_error_functions()
    {
        // erf
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::erf(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = erf(in);
                detail::load_batch(ref, expected, i);
                INFO("erf");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // erfc
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::erfc(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = erfc(in);
                detail::load_batch(ref, expected, i);
                INFO("erfc");
                CHECK_BATCH_EQ(ref, out);
            }
        }
    }

    void test_gamma_functions()
    {
        // tgamma
        {
            std::transform(gamma_input.cbegin(), gamma_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::tgamma(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, gamma_input, i);
                out = tgamma(in);
                detail::load_batch(ref, expected, i);
                INFO("tgamma");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // tgamma (negative input)
        {
            std::transform(gamma_neg_input.cbegin(), gamma_neg_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::tgamma(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, gamma_neg_input, i);
                out = tgamma(in);
                detail::load_batch(ref, expected, i);
                INFO("tgamma (negative input)");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // lgamma
        {
            std::transform(gamma_input.cbegin(), gamma_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::lgamma(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, gamma_input, i);
                out = lgamma(in);
                detail::load_batch(ref, expected, i);
                INFO("lgamma");
                CHECK_BATCH_EQ(ref, out);
            }
        }
#if !(XSIMD_WITH_AVX && !XSIMD_WITH_AVX2)

        // tgamma (negative input)
        {
            std::transform(gamma_neg_input.cbegin(), gamma_neg_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::lgamma(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, gamma_neg_input, i);
                out = lgamma(in);
                detail::load_batch(ref, expected, i);
                INFO("lgamma (negative input)");
                CHECK_BATCH_EQ(ref, out);
            }
        }
#endif
    }
};

TEST_CASE_TEMPLATE("[error gamma]", B, BATCH_FLOAT_TYPES)
{
    error_gamma_test<B> Test;

    SUBCASE("error")
    {
        Test.test_error_functions();
    }

    SUBCASE("gamma")
    {
        Test.test_gamma_functions();
    }
}
#endif
