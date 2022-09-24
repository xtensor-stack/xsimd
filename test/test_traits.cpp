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
struct traits_test
{
    using batch_type = B;
    using value_type = typename B::value_type;

    void test_simd_traits()
    {
        using traits_type = xsimd::simd_traits<value_type>;
        CHECK_EQ(traits_type::size, batch_type::size);
        constexpr bool same_type = std::is_same<B, typename traits_type::type>::value;
        CHECK_UNARY(same_type);
        using batch_bool_type = xsimd::batch_bool<value_type>;
        constexpr bool same_bool_type = std::is_same<batch_bool_type, typename traits_type::bool_type>::value;
        CHECK_UNARY(same_bool_type);

        using vector_traits_type = xsimd::simd_traits<std::vector<value_type>>;
        CHECK_EQ(vector_traits_type::size, 1);
        constexpr bool vec_same_type = std::is_same<typename vector_traits_type::type, std::vector<value_type>>::value;
        CHECK_UNARY(vec_same_type);
    }

    void test_revert_simd_traits()
    {
        using traits_type = xsimd::revert_simd_traits<batch_type>;
        CHECK_EQ(traits_type::size, batch_type::size);
        constexpr bool same_type = std::is_same<value_type, typename traits_type::type>::value;
        CHECK_UNARY(same_type);
    }

    void test_simd_return_type()
    {
        using rtype1 = xsimd::simd_return_type<value_type, float>;
        constexpr bool res1 = std::is_same<rtype1, xsimd::batch<float>>::value;
        CHECK_UNARY(res1);

        using rtype2 = xsimd::simd_return_type<bool, value_type>;
        constexpr bool res2 = std::is_same<rtype2, xsimd::batch_bool<value_type>>::value;
        CHECK_UNARY(res2);
    }
};

TEST_CASE_TEMPLATE("[traits]", B, BATCH_TYPES)
{
    traits_test<B> Test;

    SUBCASE("simd_traits")
    {
        Test.test_simd_traits();
    }

    SUBCASE("revert_simd_traits")
    {
        Test.test_revert_simd_traits();
    }

    SUBCASE("simd_return_type")
    {
        Test.test_simd_return_type();
    }
}

template <class B>
struct complex_traits_test
{
    using batch_type = B;
    using value_type = typename B::value_type;

    void test_simd_traits()
    {
        using traits_type = xsimd::simd_traits<value_type>;
        CHECK_EQ(traits_type::size, batch_type::size);
        constexpr bool same_type = std::is_same<B, typename traits_type::type>::value;
        CHECK_UNARY(same_type);
        using batch_bool_type = xsimd::batch_bool<typename value_type::value_type>;
        constexpr bool same_bool_type = std::is_same<batch_bool_type, typename traits_type::bool_type>::value;
        CHECK_UNARY(same_bool_type);

        using vector_traits_type = xsimd::simd_traits<std::vector<value_type>>;
        CHECK_EQ(vector_traits_type::size, 1);
        constexpr bool vec_same_type = std::is_same<typename vector_traits_type::type, std::vector<value_type>>::value;
        CHECK_UNARY(vec_same_type);
    }

    void test_revert_simd_traits()
    {
        using traits_type = xsimd::revert_simd_traits<batch_type>;
        CHECK_EQ(traits_type::size, batch_type::size);
        constexpr bool same_type = std::is_same<value_type, typename traits_type::type>::value;
        CHECK_UNARY(same_type);
    }

    void test_simd_return_type()
    {
        using rtype1 = xsimd::simd_return_type<value_type, float>;
        constexpr bool res1 = std::is_same<rtype1, xsimd::batch<std::complex<float>>>::value;
        CHECK_UNARY(res1);

        using rtype2 = xsimd::simd_return_type<bool, value_type>;
        constexpr bool res2 = std::is_same<rtype2, xsimd::batch_bool<typename value_type::value_type>>::value;
        CHECK_UNARY(res2);
    }
};

TEST_CASE_TEMPLATE("[complex traits]", B, BATCH_COMPLEX_TYPES)
{
    complex_traits_test<B> Test;

    SUBCASE("simd_traits")
    {
        Test.test_simd_traits();
    }

    SUBCASE("revert_simd_traits")
    {
        Test.test_revert_simd_traits();
    }

    SUBCASE("simd_return_type")
    {
        Test.test_simd_return_type();
    }
}
#endif
