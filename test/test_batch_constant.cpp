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
struct constant_batch_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    using arch_type = typename B::arch_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using bool_array_type = std::array<bool, size>;
    using batch_bool_type = typename batch_type::batch_bool_type;

    struct generator
    {
        static constexpr value_type get(size_t index, size_t /*size*/)
        {
            return index % 2 ? 0 : 1;
        }
    };

    void test_init_from_generator() const
    {
        array_type expected;
        size_t i = 0;
        std::generate(expected.begin(), expected.end(),
                      [&i]()
                      { return generator::get(i++, size); });
        constexpr auto b = xsimd::make_batch_constant<value_type, generator, arch_type>();
        INFO("batch(value_type)");
        CHECK_BATCH_EQ((batch_type)b, expected);
    }

    void test_cast() const
    {
        constexpr auto cst_b = xsimd::make_batch_constant<value_type, generator, arch_type>();
        auto b0 = cst_b.as_batch();
        auto b1 = (batch_type)cst_b;
        CHECK_BATCH_EQ(b0, b1);
        // The actual values are already tested in test_init_from_generator
    }

    struct arange
    {
        static constexpr value_type get(size_t index, size_t /*size*/)
        {
            return static_cast<value_type>(index);
        }
    };

    void test_init_from_generator_arange() const
    {
        array_type expected;
        size_t i = 0;
        std::generate(expected.begin(), expected.end(),
                      [&i]()
                      { return arange::get(i++, size); });
        constexpr auto b = xsimd::make_batch_constant<value_type, arange, arch_type>();
        INFO("batch(value_type)");
        CHECK_BATCH_EQ((batch_type)b, expected);
    }

    template <value_type V>
    struct constant
    {
        static constexpr value_type get(size_t /*index*/, size_t /*size*/)
        {
            return V;
        }
    };

    void test_init_from_constant() const
    {
        array_type expected;
        std::fill(expected.begin(), expected.end(), constant<3>::get(0, 0));
        constexpr auto b = xsimd::make_batch_constant<value_type, constant<3>, arch_type>();
        INFO("batch(value_type)");
        CHECK_BATCH_EQ((batch_type)b, expected);
    }

    void test_ops() const
    {
        constexpr auto n12 = xsimd::make_batch_constant<value_type, constant<12>, arch_type>();
        constexpr auto n3 = xsimd::make_batch_constant<value_type, constant<3>, arch_type>();

        constexpr auto n12_add_n3 = n12 + n3;
        constexpr auto n15 = xsimd::make_batch_constant<value_type, constant<15>, arch_type>();
        static_assert(std::is_same<decltype(n12_add_n3), decltype(n15)>::value, "n12 + n3 == n15");

        constexpr auto n12_sub_n3 = n12 - n3;
        constexpr auto n9 = xsimd::make_batch_constant<value_type, constant<9>, arch_type>();
        static_assert(std::is_same<decltype(n12_sub_n3), decltype(n9)>::value, "n12 - n3 == n9");

        constexpr auto n12_mul_n3 = n12 * n3;
        constexpr auto n36 = xsimd::make_batch_constant<value_type, constant<36>, arch_type>();
        static_assert(std::is_same<decltype(n12_mul_n3), decltype(n36)>::value, "n12 * n3 == n36");

        constexpr auto n12_div_n3 = n12 / n3;
        constexpr auto n4 = xsimd::make_batch_constant<value_type, constant<4>, arch_type>();
        static_assert(std::is_same<decltype(n12_div_n3), decltype(n4)>::value, "n12 / n3 == n4");

        constexpr auto n12_mod_n3 = n12 % n3;
        constexpr auto n0 = xsimd::make_batch_constant<value_type, constant<0>, arch_type>();
        static_assert(std::is_same<decltype(n12_mod_n3), decltype(n0)>::value, "n12 % n3 == n0");

        constexpr auto n12_land_n3 = n12 & n3;
        static_assert(std::is_same<decltype(n12_land_n3), decltype(n0)>::value, "n12 & n3 == n0");

        constexpr auto n12_lor_n3 = n12 | n3;
        static_assert(std::is_same<decltype(n12_lor_n3), decltype(n15)>::value, "n12 | n3 == n15");

        constexpr auto n12_lxor_n3 = n12 ^ n3;
        static_assert(std::is_same<decltype(n12_lxor_n3), decltype(n15)>::value, "n12 ^ n3 == n15");

        constexpr auto n12_uadd = +n12;
        static_assert(std::is_same<decltype(n12_uadd), decltype(n12)>::value, "+n12 == n12");

        constexpr auto n12_inv = ~n12;
        constexpr auto n12_inv_ = xsimd::make_batch_constant<value_type, constant<(value_type)~12>, arch_type>();
        static_assert(std::is_same<decltype(n12_inv), decltype(n12_inv_)>::value, "~n12 == n12_inv");

        constexpr auto n12_usub = -n12;
        constexpr auto n12_usub_ = xsimd::make_batch_constant<value_type, constant<(value_type)-12>, arch_type>();
        static_assert(std::is_same<decltype(n12_usub), decltype(n12_usub_)>::value, "-n12 == n12_usub");
    }
};

TEST_CASE_TEMPLATE("[constant batch]", B, BATCH_INT_TYPES)
{
    constant_batch_test<B> Test;
    SUBCASE("init_from_generator") { Test.test_init_from_generator(); }

    SUBCASE("as_batch") { Test.test_cast(); }

    SUBCASE("init_from_generator_arange")
    {
        Test.test_init_from_generator_arange();
    }

    SUBCASE("init_from_constant") { Test.test_init_from_constant(); }

    SUBCASE("operators")
    {
        Test.test_ops();
    }
}

template <class B>
struct constant_bool_batch_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    using arch_type = typename B::arch_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using bool_array_type = std::array<bool, size>;
    using batch_bool_type = typename batch_type::batch_bool_type;

    struct generator
    {
        static constexpr bool get(size_t index, size_t /*size*/)
        {
            return index % 2;
        }
    };

    void test_init_from_generator() const
    {
        bool_array_type expected;
        size_t i = 0;
        std::generate(expected.begin(), expected.end(),
                      [&i]()
                      { return generator::get(i++, size); });
        constexpr auto b = xsimd::make_batch_bool_constant<value_type, generator, arch_type>();
        INFO("batch_bool_constant(value_type)");
        CHECK_BATCH_EQ((batch_bool_type)b, expected);
    }

    struct split
    {
        static constexpr bool get(size_t index, size_t size)
        {
            return index < size / 2;
        }
    };

    void test_init_from_generator_split() const
    {
        bool_array_type expected;
        size_t i = 0;
        std::generate(expected.begin(), expected.end(),
                      [&i]()
                      { return split::get(i++, size); });
        constexpr auto b = xsimd::make_batch_bool_constant<value_type, split, arch_type>();
        INFO("batch_bool_constant(value_type)");
        CHECK_BATCH_EQ((batch_bool_type)b, expected);
    }

    struct inv_split
    {
        static constexpr bool get(size_t index, size_t size)
        {
            return !split().get(index, size);
        }
    };

    template <bool Val>
    struct constant
    {
        static constexpr bool get(size_t /*index*/, size_t /*size*/)
        {
            return Val;
        }
    };

    void test_cast() const
    {
        constexpr auto all_true = xsimd::make_batch_bool_constant<value_type, constant<true>, arch_type>();
        auto b0 = all_true.as_batch_bool();
        auto b1 = (batch_bool_type)all_true;
        CHECK_BATCH_EQ(b0, batch_bool_type(true));
        CHECK_BATCH_EQ(b1, batch_bool_type(true));
    }

    void test_ops() const
    {
        constexpr auto all_true = xsimd::make_batch_bool_constant<value_type, constant<true>, arch_type>();
        constexpr auto all_false = xsimd::make_batch_bool_constant<value_type, constant<false>, arch_type>();

        constexpr auto x = xsimd::make_batch_bool_constant<value_type, split, arch_type>();
        constexpr auto y = xsimd::make_batch_bool_constant<value_type, inv_split, arch_type>();

        constexpr auto x_or_y = x | y;
        static_assert(std::is_same<decltype(x_or_y), decltype(all_true)>::value, "x | y == true");

        constexpr auto x_lor_y = x || y;
        static_assert(std::is_same<decltype(x_lor_y), decltype(all_true)>::value, "x || y == true");

        constexpr auto x_and_y = x & y;
        static_assert(std::is_same<decltype(x_and_y), decltype(all_false)>::value, "x & y == false");

        constexpr auto x_land_y = x && y;
        static_assert(std::is_same<decltype(x_land_y), decltype(all_false)>::value, "x && y == false");

        constexpr auto x_xor_y = x ^ y;
        static_assert(std::is_same<decltype(x_xor_y), decltype(all_true)>::value, "x ^ y == true");

        constexpr auto not_x = !x;
        static_assert(std::is_same<decltype(not_x), decltype(y)>::value, "!x == y");

        constexpr auto inv_x = ~x;
        static_assert(std::is_same<decltype(inv_x), decltype(y)>::value, "~x == y");
    }
};

TEST_CASE_TEMPLATE("[constant bool batch]", B, BATCH_INT_TYPES)
{
    constant_bool_batch_test<B> Test;
    SUBCASE("init_from_generator") { Test.test_init_from_generator(); }

    SUBCASE("as_batch") { Test.test_cast(); }

    SUBCASE("init_from_generator_split")
    {
        Test.test_init_from_generator_split();
    }
    SUBCASE("operators")
    {
        Test.test_ops();
    }
}
#endif
