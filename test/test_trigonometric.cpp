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
struct trigonometric_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type input;
    vector_type ainput;
    vector_type atan_input;
    vector_type expected;

    trigonometric_test()
    {
        nb_input = size * 10000;
        input.resize(nb_input);
        ainput.resize(nb_input);
        atan_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(0.) + i * value_type(80.) / nb_input;
            ainput[i] = value_type(-1.) + value_type(2.) * i / nb_input;
            atan_input[i] = value_type(-10.) + i * value_type(20.) / nb_input;
        }
        expected.resize(nb_input);
    }

    void test_trigonometric_functions()
    {
        // sin
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::sin(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = sin(in);
                detail::load_batch(ref, expected, i);
                INFO("sin");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // cos
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::cos(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = cos(in);
                detail::load_batch(ref, expected, i);
                INFO("cos");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // sincos
        {
            vector_type expected2(nb_input);
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::sin(v); });
            std::transform(input.cbegin(), input.cend(), expected2.begin(),
                           [](const value_type& v)
                           { return std::cos(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out1, out2, ref1, ref2;
                detail::load_batch(in, input, i);
                std::tie(out1, out2) = sincos(in);
                detail::load_batch(ref1, expected, i);
                INFO("sincos / sin");
                CHECK_BATCH_EQ(ref1, out1);
                detail::load_batch(ref2, expected2, i);
                INFO("sincos / cos");
                CHECK_BATCH_EQ(ref2, out2);
            }
        }
        // tan
        {
            std::transform(input.cbegin(), input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::tan(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, input, i);
                out = tan(in);
                detail::load_batch(ref, expected, i);
                INFO("tan");
                CHECK_BATCH_EQ(ref, out);
            }
        }
    }

    void test_reciprocal_functions()
    {

        // asin
        {
            std::transform(ainput.cbegin(), ainput.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::asin(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, ainput, i);
                out = asin(in);
                detail::load_batch(ref, expected, i);
                INFO("asin");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // acos
        {
            std::transform(ainput.cbegin(), ainput.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::acos(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, ainput, i);
                out = acos(in);
                detail::load_batch(ref, expected, i);
                INFO("acos");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // atan
        {
            std::transform(atan_input.cbegin(), atan_input.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::atan(v); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, out, ref;
                detail::load_batch(in, atan_input, i);
                out = atan(in);
                detail::load_batch(ref, expected, i);
                INFO("atan");
                CHECK_BATCH_EQ(ref, out);
            }
        }
        // atan2
        {
            std::transform(atan_input.cbegin(), atan_input.cend(), input.cbegin(), expected.begin(),
                           [](const value_type& v, const value_type& r)
                           { return std::atan2(v, r); });
            for (size_t i = 0; i < nb_input; i += size)
            {
                batch_type in, rhs, out, ref;
                detail::load_batch(in, atan_input, i);
                detail::load_batch(rhs, input, i);
                out = atan2(in, rhs);
                detail::load_batch(ref, expected, i);
                INFO("atan2");
                CHECK_BATCH_EQ(ref, out);
            }
        }
    }
};

TEST_CASE_TEMPLATE("[trigonometric]", B, BATCH_FLOAT_TYPES)
{
    trigonometric_test<B> Test;
    SUBCASE("trigonometric")
    {
        Test.test_trigonometric_functions();
    }

    SUBCASE("reciprocal")
    {
        Test.test_reciprocal_functions();
    }
}
#ifndef __FAST_MATH__
TEST_CASE_TEMPLATE("[atan2 signed zero pairs]", B, BATCH_FLOAT_TYPES)
{
    using value_type = typename B::value_type;
    const value_type zero(0);
    const std::array<std::pair<value_type, value_type>, 20> samples = {
        std::pair { zero, zero }, std::pair { -zero, zero },
        std::pair { zero, -zero }, std::pair { -zero, -zero },
        std::pair { zero, value_type(1) }, std::pair { -zero, value_type(1) },
        std::pair { zero, value_type(-1) }, std::pair { -zero, value_type(-1) },
        std::pair { value_type(1), zero }, std::pair { value_type(-1), zero },
        std::pair { value_type(1), -zero }, std::pair { value_type(-1), -zero },
        std::pair { value_type(1), value_type(1) }, std::pair { value_type(-1), value_type(1) },
        std::pair { value_type(1), value_type(-1) }, std::pair { value_type(-1), value_type(-1) },
        std::pair { value_type(3), value_type(4) }, std::pair { value_type(-3), value_type(4) },
        std::pair { value_type(3), value_type(-4) }, std::pair { value_type(-3), value_type(-4) }
    };
    std::array<value_type, B::size> ys, xs, results;
    const auto tolerance = value_type(64) * std::numeric_limits<value_type>::epsilon();
    for (size_t start = 0; start < samples.size(); ++start)
    {
        for (size_t lane = 0; lane < B::size; ++lane)
        {
            const auto sample = samples[(start + lane) % samples.size()];
            ys[lane] = sample.first;
            xs[lane] = sample.second;
        }
        xsimd::atan2(B::load_unaligned(ys.data()), B::load_unaligned(xs.data())).store_unaligned(results.data());
        for (size_t lane = 0; lane < B::size; ++lane)
        {
            const auto expected = std::atan2(ys[lane], xs[lane]);
            CHECK_EQ(results[lane], doctest::Approx(expected).epsilon(tolerance).scale(1));
            CHECK_EQ(std::signbit(results[lane]), std::signbit(expected));
            if (expected == zero)
            {
                CHECK_EQ(results[lane], expected);
            }
        }
    }
}
#if !defined(XSIMD_NO_INFINITIES) && !defined(XSIMD_NO_NANS)
TEST_CASE_TEMPLATE("[atan2 infinite argument pairs]", B, BATCH_FLOAT_TYPES)
{
    using value_type = typename B::value_type;
    const value_type zero(0);
    const auto inf = std::numeric_limits<value_type>::infinity();
    const auto nan = std::numeric_limits<value_type>::quiet_NaN();
    const std::array<std::pair<value_type, value_type>, 20> samples = {
        std::pair { inf, inf }, std::pair { -inf, inf },
        std::pair { inf, -inf }, std::pair { -inf, -inf },
        std::pair { inf, value_type(1) }, std::pair { -inf, value_type(1) },
        std::pair { inf, value_type(-1) }, std::pair { -inf, value_type(-1) },
        std::pair { value_type(1), inf }, std::pair { value_type(-1), inf },
        std::pair { value_type(1), -inf }, std::pair { value_type(-1), -inf },
        std::pair { zero, inf }, std::pair { -zero, inf },
        std::pair { zero, -inf }, std::pair { -zero, -inf },
        std::pair { nan, value_type(1) }, std::pair { value_type(1), nan },
        std::pair { value_type(1), value_type(1) }, std::pair { value_type(-1), value_type(-1) }
    };
    std::array<value_type, B::size> ys, xs, results;
    const auto tolerance = value_type(64) * std::numeric_limits<value_type>::epsilon();
    for (size_t start = 0; start < samples.size(); ++start)
    {
        for (size_t lane = 0; lane < B::size; ++lane)
        {
            const auto sample = samples[(start + lane) % samples.size()];
            ys[lane] = sample.first;
            xs[lane] = sample.second;
        }
        xsimd::atan2(B::load_unaligned(ys.data()), B::load_unaligned(xs.data())).store_unaligned(results.data());
        for (size_t lane = 0; lane < B::size; ++lane)
        {
            const auto expected = std::atan2(ys[lane], xs[lane]);
            if (std::isnan(expected))
            {
                CHECK(std::isnan(results[lane]));
                continue;
            }
            CHECK(std::isfinite(results[lane]));
            CHECK_EQ(results[lane], doctest::Approx(expected).epsilon(tolerance).scale(1));
            CHECK_EQ(std::signbit(results[lane]), std::signbit(expected));
            if (expected == zero)
            {
                CHECK_EQ(results[lane], zero);
            }
        }
    }
}
#endif
#endif
#endif
