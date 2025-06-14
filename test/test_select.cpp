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
struct select_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    using arch_type = typename B::arch_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type lhs_input;
    vector_type rhs_input;
    vector_type expected;
    vector_type res;

    select_test()
    {
        nb_input = size * 10000;
        lhs_input.resize(nb_input);
        rhs_input.resize(nb_input);
        auto clamp = [](double v)
        {
            return static_cast<value_type>(std::min(v, static_cast<double>(std::numeric_limits<value_type>::max())));
        };
        for (size_t i = 0; i < nb_input; ++i)
        {
            lhs_input[i] = clamp(i / 4 + 1.2 * std::sqrt(i + 0.25));
            rhs_input[i] = clamp(10.2 / (i + 2) + 0.25);
        }
        expected.resize(nb_input);
        res.resize(nb_input);
    }

    void test_select_dynamic()
    {
        for (size_t i = 0; i < nb_input; ++i)
        {
            expected[i] = lhs_input[i] > value_type(3) ? lhs_input[i] : rhs_input[i];
        }

        batch_type lhs_in, rhs_in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(lhs_in, lhs_input, i);
            detail::load_batch(rhs_in, rhs_input, i);
            out = xsimd::select(lhs_in > value_type(3), lhs_in, rhs_in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }
    struct pattern
    {
        static constexpr bool get(std::size_t i, std::size_t) { return i % 2; }
    };

    void test_select_static()
    {
        constexpr auto mask = xsimd::make_batch_bool_constant<value_type, pattern, arch_type>();

        for (size_t i = 0; i < nb_input; ++i)
        {
            expected[i] = mask.get(i % size) ? lhs_input[i] : rhs_input[i];
        }

        batch_type lhs_in, rhs_in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(lhs_in, lhs_input, i);
            detail::load_batch(rhs_in, rhs_input, i);
            out = xsimd::select(mask, lhs_in, rhs_in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff(res, expected);
        CHECK_EQ(diff, 0);
    }
};

TEST_CASE_TEMPLATE("[select]", B, BATCH_TYPES)
{
    select_test<B> Test;
    SUBCASE("select_dynamic") { Test.test_select_dynamic(); }
    SUBCASE("select_static") { Test.test_select_static(); }
}
#endif
