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
struct complex_trigonometric_test
{
    using batch_type = B;
    using real_batch_type = typename B::real_batch;
    using value_type = typename B::value_type;
    using real_value_type = typename value_type::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;

    size_t nb_input;
    vector_type input;
    vector_type ainput;
    vector_type atan_input;
    vector_type expected;

    complex_trigonometric_test()
    {
        nb_input = size * 10000;
        input.resize(nb_input);
        ainput.resize(nb_input);
        atan_input.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            input[i] = value_type(real_value_type(0.) + i * real_value_type(80.) / nb_input,
                                  real_value_type(0.1) + i * real_value_type(56.) / nb_input);
            ainput[i] = value_type(real_value_type(-1.) + real_value_type(2.) * i / nb_input,
                                   real_value_type(-1.1) + real_value_type(2.1) * i / nb_input);
            // Avoid sampling the branch cut of complex atan at (Re=0, |Im|>=1):
            // the sign of Re(catan(+/-0 + i*y)) for |y|>1 is implementation-defined
            // (C99 7.3.4.1), and glibc and musl disagree there. Starting Re at -9.5
            // makes Re=0 occur at i=19/40 of the range, where |Im|<1 (analytic).
            atan_input[i] = value_type(real_value_type(-9.5) + i * real_value_type(20.) / nb_input,
                                       real_value_type(-9.) + i * real_value_type(21.) / nb_input);
        }
        expected.resize(nb_input);
    }

    void test_sin()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::sin; return sin(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, input, i);
            out = sin(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }

    void test_cos()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::cos; return cos(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, input, i);
            out = cos(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }

    void test_sincos()
    {
        vector_type expected2(nb_input), res2(nb_input);
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::sin; return sin(v); });
        std::transform(input.cbegin(), input.cend(), expected2.begin(),
                       [](const value_type& v)
                       { using std::cos; return cos(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out1, out2, ref1, ref2;
            detail::load_batch(in, input, i);
            std::tie(out1, out2) = sincos(in);

            detail::load_batch(ref1, expected, i);
            CHECK_BATCH_EQ(ref1, out1);

            detail::load_batch(ref2, expected2, i);
            CHECK_BATCH_EQ(ref2, out2);
        }
    }

    void test_tan()
    {
        test_conditional_tan<real_value_type>();
    }

    void test_asin()
    {
        std::transform(ainput.cbegin(), ainput.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::asin; return asin(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, ainput, i);
            out = asin(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }

    void test_acos()
    {
        std::transform(ainput.cbegin(), ainput.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::acos; return acos(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, ainput, i);
            out = acos(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }

    void test_atan()
    {
        std::transform(atan_input.cbegin(), atan_input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::atan; return atan(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, atan_input, i);
            out = atan(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }

private:
    void test_tan_impl()
    {
        std::transform(input.cbegin(), input.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::tan; return tan(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, input, i);
            out = tan(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }

    template <class T, std::enable_if_t<!std::is_same_v<T, float>, int> = 0>
    void test_conditional_tan()
    {
        test_tan_impl();
    }

    template <class T, std::enable_if_t<std::is_same_v<T, float>, int> = 0>
    void test_conditional_tan()
    {
#if (XSIMD_ARM_INSTR_SET >= XSIMD_ARM7_NEON_VERSION)
#if DEBUG_ACCURACY
        test_tan_impl();
#endif
#else
        test_tan_impl();
#endif
    }
};

TEST_CASE_TEMPLATE("[complex trigonometric]", B, BATCH_COMPLEX_TYPES)
{
    complex_trigonometric_test<B> Test;
    SUBCASE("sin")
    {
        Test.test_sin();
    }

    SUBCASE("cos")
    {
        Test.test_cos();
    }

    SUBCASE("sincos")
    {
        Test.test_sincos();
    }

    SUBCASE("tan")
    {
        Test.test_tan();
    }

    SUBCASE("asin")
    {
        Test.test_asin();
    }

    SUBCASE("acos")
    {
        Test.test_acos();
    }

    SUBCASE("atan")
    {
        Test.test_atan();
    }
}

// Cases of issue 1432 that the fix changed: each gave a wrong value or a wrong sign of zero before it. The expected
// values are the exact f(z) rounded to double; cond is max(1, |z f'(z) / f(z)|) (see detail::check_complex_cases).
TEST_CASE_TEMPLATE("[complex trigonometric issue 1432]", B, BATCH_COMPLEX_TYPES)
{
    using xsimd::constants::pi;
    using xsimd::constants::pio2;

    SUBCASE("asin")
    {
        const std::vector<std::complex<double>> inputs = {
            { -2, -0. }, // the -0 sets the sign of Im asin(z)
            { -0., -0. }, // signed zeros
            { 1, 800 }, // large imaginary part
            { 10000, 1 }, // large real part
        };
        const std::vector<std::complex<double>> expected = {
            { -pio2<double>(), -1.3169578969248168 },
            { -0., -0. },
            { 0.0012499983724006398, 7.377760080100202 },
            { 1.57069632679473, 9.903487555036127 },
        };
        const std::vector<double> cond = { 1, 1, 1, 1 };
        CHECK_COMPLEX_CASES(B, asin, inputs, expected, cond);
    }

    SUBCASE("acos")
    {
        const std::vector<std::complex<double>> inputs = {
            { -2, -0. }, // the -0 sets the sign of Im acos(z)
            { 0., -0. }, // signed zeros
            { 1, 800 }, // large imaginary part
            { 10000, 1 }, // large real part
        };
        const std::vector<std::complex<double>> expected = {
            { pi<double>(), 1.3169578969248168 },
            { pio2<double>(), 0. },
            { 1.569546328422496, -7.377760080100202 },
            { 0.00010000000016666667, -9.903487555036127 },
        };
        const std::vector<double> cond = { 1, 1, 1, 1 };
        CHECK_COMPLEX_CASES(B, acos, inputs, expected, cond);
    }

    SUBCASE("tan")
    {
        const std::vector<std::complex<double>> inputs = {
            { 1.5707f, 0. }, // near the pole pi/2 = 1.570796
            { 2, -500 }, // large imaginary part
        };
        const std::vector<std::complex<double>> expected = {
            { 10386.654119664716, 0. },
            { -0., -1. },
        };
        const std::vector<double> cond = { 1.63e4, 1 };
        CHECK_COMPLEX_CASES(B, tan, inputs, expected, cond);
    }

    SUBCASE("asinh")
    {
        const std::vector<std::complex<double>> inputs = {
            { -0., -3 }, // the -0 sets the sign of Re asinh(z)
            { -0., -0. }, // signed zeros
            { 2, 0. }, // real axis
            { 1, -500 }, // large imaginary part
            { 10000, 1 }, // large real part
        };
        const std::vector<std::complex<double>> expected = {
            { -1.762747174039086, -pio2<double>() },
            { -0., -0. },
            { 1.4436354751788103, 0. },
            { 6.907756278988637, -1.5687963254615769 },
            { 9.903487560036128, 9.999999916666668e-05 },
        };
        const std::vector<double> cond = { 1, 1, 1, 1, 1 };
        CHECK_COMPLEX_CASES(B, asinh, inputs, expected, cond);
    }

    SUBCASE("acosh")
    {
        const std::vector<std::complex<double>> inputs = {
            { -4, -3 }, // x < 0, y < 0
            { -2, 0. }, // the +0 sets the sign of Im acosh(z)
            { 1, -0. }, // signed zeros
            { 1, 800 }, // large imaginary part
            { 10000, 1 }, // large real part
        };
        const std::vector<std::complex<double>> expected = {
            { 2.2999140408792695, -2.488413180146375 },
            { 1.3169578969248168, pi<double>() },
            { 0., -0. },
            { 7.377760080100202, 1.569546328422496 },
            { 9.903487555036127, 0.00010000000016666667 },
        };
        const std::vector<double> cond = { 1, 1, 1, 1, 1 };
        CHECK_COMPLEX_CASES(B, acosh, inputs, expected, cond);
    }

    SUBCASE("tanh")
    {
        const std::vector<std::complex<double>> inputs = {
            { -400, 2 }, // large real part
            { 0., -10.9958f }, // near the pole -7 pi/2 i = -10.995574 i
        };
        const std::vector<std::complex<double>> expected = {
            { -1., -0. },
            { 0., 4430.056602996832 },
        };
        const std::vector<double> cond = { 1, 4.87e4 };
        CHECK_COMPLEX_CASES(B, tanh, inputs, expected, cond);
    }
}
#endif
