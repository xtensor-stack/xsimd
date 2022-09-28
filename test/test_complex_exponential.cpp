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
struct complex_exponential_test
{
    using batch_type = B;
    using real_batch_type = typename B::real_batch;
    using value_type = typename B::value_type;
    using real_value_type = typename value_type::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type exp_input;
    vector_type huge_exp_input;
    vector_type log_input;
    vector_type expected;
    vector_type res;

    complex_exponential_test()
    {
        nb_input = 10000 * size;
        exp_input.resize(nb_input);
        huge_exp_input.resize(nb_input);
        log_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            exp_input[i] = value_type(real_value_type(-1.5) + i * real_value_type(3) / nb_input,
                                      real_value_type(-1.3) + i * real_value_type(2) / nb_input);
            huge_exp_input[i] = value_type(real_value_type(0), real_value_type(102.12) + i * real_value_type(100.) / nb_input);
            log_input[i] = value_type(real_value_type(0.001 + i * 100 / nb_input),
                                      real_value_type(0.002 + i * 110 / nb_input));
        }
        expected.resize(nb_input);
        res.resize(nb_input);
    }

    void test_exp()
    {
        std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::exp; return exp(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, exp_input, i);
            out = exp(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }

    void test_expm1()
    {
        std::transform(exp_input.cbegin(), exp_input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using xsimd::expm1; return expm1(v); });

        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, exp_input, i);
            out = expm1(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }

    void test_huge_exp()
    {
        std::transform(huge_exp_input.cbegin(), huge_exp_input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::exp; return exp(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, huge_exp_input, i);
            out = exp(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }

    void test_log()
    {
        std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::log; return log(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, log_input, i);
            out = log(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }

    void test_log2()
    {
        std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using xsimd::log2; return log2(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, log_input, i);
            out = log2(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }

    void test_log10()
    {
        std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::log10; return log10(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, log_input, i);
            out = log10(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }

    void test_log1p()
    {
        std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using xsimd::log1p; return log1p(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, log_input, i);
            out = log1p(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }

    void test_sign()
    {
        std::transform(log_input.cbegin(), log_input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using xsimd::sign; return sign(v); });
        batch_type in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(in, log_input, i);
            out = sign(in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }
};

TEST_CASE_TEMPLATE("[complex exponential]", B, BATCH_COMPLEX_TYPES)
{
    complex_exponential_test<B> Test;

    SUBCASE("exp")
    {
        Test.test_exp();
    }

    SUBCASE("expm1")
    {
        Test.test_expm1();
    }

    SUBCASE("huge_exp")
    {
        Test.test_huge_exp();
    }

    SUBCASE("log")
    {
        Test.test_log();
    }

    SUBCASE("log2")
    {
        Test.test_log2();
    }

    SUBCASE("log10")
    {
        Test.test_log10();
    }

    SUBCASE("log1p")
    {
        Test.test_log1p();
    }

    SUBCASE("sign")
    {
        Test.test_sign();
    }
}
#endif
