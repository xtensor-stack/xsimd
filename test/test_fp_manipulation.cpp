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
struct fp_manipulation_test
{
    using batch_type = B;
    using arch_type = typename B::arch_type;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using int_value_type = xsimd::as_integer_t<value_type>;
    using int_batch_type = xsimd::batch<int_value_type, arch_type>;

    array_type input;
    int_value_type exponent;

    fp_manipulation_test()
    {
        exponent = 5;
        for (size_t i = 0; i < size; ++i)
        {
            input[i] = value_type(i) / 4 + value_type(1.2) * std::sqrt(value_type(i + 0.25));
        }
    }

    void test_fp_manipulations() const
    {
        int_batch_type bexp(exponent);
        // ldexp
        {
            array_type expected;
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [this](const value_type& v)
                           { return std::ldexp(v, exponent); });
            batch_type res = xsimd::ldexp(batch_input(), bexp);
            INFO("ldexp");
            CHECK_BATCH_EQ(res, expected);
        }
        // frexp
        {
            array_type expected;
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { int tmp; return std::frexp(v, &tmp); });
            batch_type res = xsimd::frexp(batch_input(), bexp);
            INFO("frexp");
            CHECK_BATCH_EQ(res, expected);
        }
    }

private:
    batch_type batch_input() const
    {
        return batch_type::load_unaligned(input.data());
    }
};

TEST_CASE_TEMPLATE("[fp manipulation]", B, BATCH_FLOAT_TYPES)
{
    fp_manipulation_test<B> Test;
    Test.test_fp_manipulations();
}
#endif
