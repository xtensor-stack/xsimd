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
struct batch_float_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using bool_array_type = std::array<bool, size>;

    array_type lhs;
    array_type rhs;

    batch_float_test()
    {
        for (size_t i = 0; i < size; ++i)
        {
            lhs[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25));
            if (lhs[i] == value_type(0))
            {
                lhs[i] += value_type(0.1);
            }
            rhs[i] = value_type(10.2) / (i + 2) + value_type(0.25);
        }
    }

    void test_reciprocal() const
    {
        // reciprocal
        {
            array_type res, expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& l)
                           { return value_type(1) / l; });
            batch_type res1 = reciprocal(batch_lhs());
            res1.store_unaligned(res.data());
            size_t diff = detail::get_nb_diff_near(res, expected, 1e-12f);
            INFO("reciprocal");
            CHECK_EQ(diff, 0);
        }
    }

    void test_rsqrt() const
    {
        // rsqrt
        {
            array_type res, expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& l)
                           { return std::ceil((value_type(1) / std::sqrt(l)) * value_type(100)); });
            batch_type res1 = ceil(rsqrt(batch_lhs()) * value_type(100));
            res1.store_unaligned(res.data());
            size_t diff = detail::get_nb_diff_near(res, expected, 1.5f * std::pow(2, 12));
            INFO("rsqrt");
            CHECK_EQ(diff, 0);
        }
    }

    void test_sqrt() const
    {
        // sqrt
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& l)
                           { return std::sqrt(l); });
            batch_type res = sqrt(batch_lhs());
            INFO("sqrt");
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_haddp() const
    {
        batch_type haddp_input[size];
        for (size_t i = 0; i < size; i += 2)
        {
            haddp_input[i] = batch_lhs();
            if (i + 1 < size)
            {
                haddp_input[i + 1] = batch_rhs();
            }
        }
        array_type expected;
        std::fill(expected.begin(), expected.end(), value_type(0));
        for (size_t i = 0; i < size; ++i)
        {
            for (size_t j = 0; j < size; j += 2)
            {
                expected[j] += lhs[i];
                if (j + 1 < size)
                {
                    expected[j + 1] += rhs[i];
                }
            }
        }
        auto res = haddp(haddp_input);
        INFO("haddp");
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
};

TEST_CASE_TEMPLATE("[xsimd batch float]", B, BATCH_FLOAT_TYPES)
{
    batch_float_test<B> Test;

    SUBCASE("reciprocal") { Test.test_reciprocal(); }

    SUBCASE("sqrt") { Test.test_sqrt(); }

    SUBCASE("rsqrt") { Test.test_rsqrt(); }

    SUBCASE("haddp") { Test.test_haddp(); }
}
#endif
