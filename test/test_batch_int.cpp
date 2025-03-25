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

#include <climits>

namespace xsimd
{
    template <class T, std::size_t N = T::size>
    struct test_int_min_max
    {
        bool run()
        {
            return true;
        }
    };

    template <class T>
    struct test_int_min_max<batch<T>, 2>
    {
        void run()
        {
            using B = batch<T>;
            using BB = batch_bool<T>;
            using A = std::array<T, 2>;

            T max = std::numeric_limits<T>::max();
            T min = std::numeric_limits<T>::min();
            std::array<T, 2> maxmin_cmp { { max, min } };
            B maxmin = { max, min };
            INFO("numeric max and min");
            CHECK_BATCH_EQ(maxmin, maxmin_cmp);

            B a = { 1, 3 };
            B b(2);
            B c = { 2, 3 };

            auto r1 = xsimd::max(a, c);
            auto r3 = xsimd::min(a, c);

            INFO("max");
            CHECK_BATCH_EQ(r1, (A { { 2, 3 } }));
            INFO("min");
            CHECK_BATCH_EQ(r3, (A { { 1, 3 } }));

            auto r4 = a < b; // test lt
            BB e4 = { 1, 0 };
            CHECK_UNARY(xsimd::all(r4 == e4));
        }
    };

    template <class T>
    struct test_int_min_max<batch<T>, 4>
    {
        void run()
        {
            using B = batch<T>;
            using BB = batch_bool<T>;
            using A = std::array<T, 4>;

            B a = { 1, 3, 1, 1 };
            B b(2);
            B c = { 2, 3, 2, 3 };

            auto r1 = xsimd::max(a, c);
            auto r3 = xsimd::min(a, c);

            INFO("max");
            CHECK_BATCH_EQ(r1, (A { { 2, 3, 2, 3 } }));
            INFO("min");
            CHECK_BATCH_EQ(r3, (A { { 1, 3, 1, 1 } }));

            auto r4 = a < b; // test lt
            BB e4 = { 1, 0, 1, 1 };
            CHECK_UNARY(xsimd::all(r4 == e4));
        }
    };

    template <class T>
    struct test_int_min_max<batch<T>, 8>
    {
        void run()
        {
            using B = batch<T>;
            using BB = batch_bool<T>;
            using A = std::array<T, 8>;

            T max = std::numeric_limits<T>::max();
            T min = std::numeric_limits<T>::min();
            std::array<T, 8> maxmin_cmp { { 0, 0, max, 0, min, 0, 0, 0 } };
            B maxmin = { 0, 0, max, 0, min, 0, 0, 0 };
            INFO("numeric max and min");
            CHECK_BATCH_EQ(maxmin, maxmin_cmp);

            B a { 1, 3, 1, 3, 1, 1, 3, 3 };
            B b { 2 };
            B c { 2, 3, 2, 3, 2, 3, 2, 3 };

            auto r1 = xsimd::max(a, c);
            auto r3 = xsimd::min(a, c);
            auto r4 = a < b; // test lt
            INFO("max");
            CHECK_BATCH_EQ(r1, (A { { 2, 3, 2, 3, 2, 3, 3, 3 } }));
            INFO("min");
            CHECK_BATCH_EQ(r3, (A { { 1, 3, 1, 3, 1, 1, 2, 3 } }));

            BB e4 = { 1, 0, 1, 0, 1, 1, 0, 0 };
            CHECK_UNARY(xsimd::all(r4 == e4));
        }
    };

    template <class T>
    struct test_int_min_max<batch<T>, 16>
    {
        void run()
        {
            using B = batch<T>;
            using BB = batch_bool<T>;
            using A = std::array<T, 16>;

            T max = std::numeric_limits<T>::max();
            T min = std::numeric_limits<T>::min();
            std::array<T, 16> maxmin_cmp { { 0, 0, max, 0, min, 0, 0, 0, 0, 0, max, 0, min, 0, 0, 0 } };
            B maxmin = { 0, 0, max, 0, min, 0, 0, 0, 0, 0, max, 0, min, 0, 0, 0 };
            INFO("numeric max and min");
            CHECK_BATCH_EQ(maxmin, maxmin_cmp);

            B a = { 1, 3, 1, 3, 1, 3, 1, 3, 3, 3, 3, 3, min, max, max, min };
            B b(2);
            B c = { 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3 };
            auto r1 = xsimd::max(a, b);
            auto r3 = xsimd::min(a, b);
            auto r4 = a < b; // test lt
            auto r5 = a == c;
            auto r6 = a != c;

            INFO("max");
            CHECK_BATCH_EQ(r1, (A { { 2, 3, 2, 3, 2, 3, 2, 3, 3, 3, 3, 3, 2, max, max, 2 } }));
            INFO("min");
            CHECK_BATCH_EQ(r3, (A { { 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, min, 2, 2, min } }));

            BB e4 = { 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1 };
            CHECK_UNARY(xsimd::all(r4 == e4));

            BB e5 = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0 };
            CHECK_UNARY(xsimd::all(r5 == e5));
            CHECK_UNARY(xsimd::all(r6 == !e5));
        }
    };

    template <class T>
    struct test_int_min_max<batch<T>, 32>
    {
        void run()
        {
            using B = batch<T>;
            using BB = batch_bool<T>;
            using A = std::array<T, 32>;
            T max = std::numeric_limits<T>::max();
            T min = std::numeric_limits<T>::min();

            B a = { 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3, 3, 3, min, max, max, min };
            B b = 2;

            auto r1 = xsimd::max(a, b);
            auto r3 = xsimd::min(a, b);
            auto r4 = a < b; // test lt
            INFO("max");
            CHECK_BATCH_EQ(r1, (A { { 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 3, 3, 3, 3, 2, max, max, 2 } }));
            INFO("min");
            CHECK_BATCH_EQ(r3, (A { { 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, min, 2, 2, min } }));

            BB e4 = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1 };
            CHECK_UNARY(xsimd::all(r4 == e4));
        }
    };
}

template <class B>
struct batch_int_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using bool_array_type = std::array<bool, size>;

    array_type lhs;
    array_type rhs;
    array_type shift;

    batch_int_test()
    {
        using signed_value_type = typename std::make_signed<value_type>::type;
        for (size_t i = 0; i < size; ++i)
        {
            bool negative_lhs = std::is_signed<value_type>::value && (i % 2 == 1);
            lhs[i] = value_type(i) * (negative_lhs ? -10 : 10);
            if (lhs[i] == value_type(0))
            {
                lhs[i] += value_type(1);
            }
            rhs[i] = value_type(i) + value_type(4);
            shift[i] = signed_value_type(i) % (CHAR_BIT * sizeof(value_type));
        }
    }

    void test_modulo() const
    {
        // batch % batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l % r; });
            batch_type res = batch_lhs() % batch_rhs();
            INFO("batch % batch");
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_shift() const
    {
        int32_t nb_sh = 3;
        // batch << scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [nb_sh](const value_type& v)
                           { return v << nb_sh; });
            batch_type res = batch_lhs() << nb_sh;
            INFO("batch << scalar");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch << batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), shift.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l << r; });
            batch_type res = batch_lhs() << batch_shift();
            INFO("batch << batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch >> scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [nb_sh](const value_type& v)
                           { return v >> nb_sh; });
            batch_type res = batch_lhs() >> nb_sh;
            INFO("batch >> scalar");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch >> batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), shift.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l >> r; });
            batch_type res = batch_lhs() >> batch_shift();
            INFO("batch >> batch");
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_more_shift() const
    {
        int32_t s = static_cast<int32_t>(sizeof(value_type) * 8);
        batch_type lhs = batch_type(value_type(1));
        batch_type res;

        for (int32_t i = 0; i < s; ++i)
        {
            res = lhs << i;
            value_type expected = value_type(1) << i;
            for (std::size_t j = 0; j < size; ++j)
            {
                CHECK_EQ(res.get(j), expected);
            }
        }
        lhs = batch_type(std::numeric_limits<value_type>::max());
        for (int32_t i = 0; i < s; ++i)
        {
            res = lhs >> i;
            value_type expected = std::numeric_limits<value_type>::max() >> i;
            for (std::size_t j = 0; j < size; ++j)
            {
                CHECK_EQ(res.get(j), expected);
            }
        }
    }

    void test_min_max() const
    {
        xsimd::test_int_min_max<batch_type> t;
        t.run();
    }

    void test_less_than_underflow() const
    {
        batch_type test_negative_compare = batch_type(5) - 6;
        if (std::is_unsigned<value_type>::value)
        {
            CHECK_FALSE(xsimd::any(test_negative_compare < 1));
        }
        else
        {
            CHECK_UNARY(xsimd::all(test_negative_compare < 1));
        }
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

    batch_type batch_shift() const
    {
        return batch_type::load_unaligned(shift.data());
    }
};

TEST_CASE_TEMPLATE("[batch int tests]", B, BATCH_INT_TYPES)
{
    batch_int_test<B> Test;

    SUBCASE("modulo")
    {
        Test.test_modulo();
    }

    SUBCASE("shift")
    {
        Test.test_shift();
    }

    SUBCASE("more_shift")
    {
        Test.test_more_shift();
    }

    SUBCASE("min_max")
    {
        Test.test_min_max();
    }

    SUBCASE("less_than_underflow")
    {
        Test.test_less_than_underflow();
    }
}
#endif
