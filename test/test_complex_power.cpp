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
struct complex_power_test
{
    using batch_type = B;
    using real_batch_type = typename B::real_batch;
    using value_type = typename B::value_type;
    using real_value_type = typename value_type::value_type;
    static constexpr size_t size = B::size;
    using vector_type = std::vector<value_type>;
    using real_vector_type = std::vector<real_value_type>;

    size_t nb_input;
    real_vector_type lhs_p;
    vector_type lhs_nn;
    vector_type lhs_pn;
    vector_type lhs_np;
    vector_type lhs_pp;
    vector_type rhs;
    vector_type expected;
    vector_type res;

    complex_power_test()
    {
        nb_input = 10000 * size;
        lhs_p.resize(nb_input);
        lhs_nn.resize(nb_input);
        lhs_pn.resize(nb_input);
        lhs_np.resize(nb_input);
        lhs_pp.resize(nb_input);
        rhs.resize(nb_input);
        for (size_t i = 0; i < nb_input; ++i)
        {
            real_value_type real = (real_value_type(i) / 4 + real_value_type(1.2) * std::sqrt(real_value_type(i + 0.25))) / 100;
            real_value_type imag = (real_value_type(i) / 7 + real_value_type(1.7) * std::sqrt(real_value_type(i + 0.37))) / 100;
            lhs_p[i] = real;
            lhs_nn[i] = value_type(-real, -imag);
            lhs_pn[i] = value_type(real, -imag);
            lhs_np[i] = value_type(-real, imag);
            lhs_pp[i] = value_type(real, imag);
            rhs[i] = value_type(real_value_type(10.2) / (i + 2) + real_value_type(0.25),
                                real_value_type(9.1) / (i + 3) + real_value_type(0.45));
        }
        expected.resize(nb_input);
        res.resize(nb_input);
    }

    void test_abs()
    {
        real_vector_type real_expected(nb_input), real_res(nb_input);
        std::transform(lhs_np.cbegin(), lhs_np.cend(), real_expected.begin(),
                       [](const value_type& v)
                       { using std::abs; return abs(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in;
            real_batch_type out, ref;
            detail::load_batch(in, lhs_np, i);
            out = abs(in);
            detail::load_batch(ref, real_expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }

    void test_arg()
    {
        real_vector_type real_expected(nb_input), real_res(nb_input);
        std::transform(lhs_np.cbegin(), lhs_np.cend(), real_expected.begin(),
                       [](const value_type& v)
                       { using std::arg; return arg(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in;
            real_batch_type out, ref;
            detail::load_batch(in, lhs_np, i);
            out = arg(in);
            detail::load_batch(ref, real_expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
    }

    void test_pow()
    {
        std::transform(lhs_np.cbegin(), lhs_np.cend(), rhs.cbegin(), expected.begin(),
                       [](const value_type& l, const value_type& r)
                       { using std::pow; return pow(l, r); });
        batch_type lhs_in, rhs_in, out;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(lhs_in, lhs_np, i);
            detail::load_batch(rhs_in, rhs, i);
            out = pow(lhs_in, rhs_in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff_near(res, expected, std::numeric_limits<real_value_type>::epsilon());
        CHECK_EQ(diff, 0);
    }

    void test_pow_real_complex()
    {
        std::transform(lhs_p.cbegin(), lhs_p.cend(), lhs_pp.cbegin(), expected.begin(),
                       [](const real_value_type& l, const value_type& r)
                       { using std::pow; return pow(l, r); });
        batch_type rhs_in, out;
        real_batch_type lhs_in;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(lhs_in, lhs_p, i);
            detail::load_batch(rhs_in, lhs_pp, i);
            out = pow(lhs_in, rhs_in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff_near(res, expected, std::numeric_limits<real_value_type>::epsilon());
        CHECK_EQ(diff, 0);
    }

    void test_pow_complex_real()
    {
        std::transform(lhs_pp.cbegin(), lhs_pp.cend(), lhs_p.cbegin(), expected.begin(),
                       [](const value_type& l, const real_value_type& r)
                       { using std::pow; return pow(l, r); });
        batch_type rhs_in, out;
        real_batch_type lhs_in;
        for (size_t i = 0; i < nb_input; i += size)
        {
            detail::load_batch(lhs_in, lhs_p, i);
            detail::load_batch(rhs_in, lhs_pp, i);
            out = pow(lhs_in, rhs_in);
            detail::store_batch(out, res, i);
        }
        size_t diff = detail::get_nb_diff_near(res, expected, std::numeric_limits<real_value_type>::epsilon());
        CHECK_EQ(diff, 0);
    }

    void test_sqrt_nn()
    {
        std::transform(lhs_nn.cbegin(), lhs_nn.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::sqrt; return sqrt(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, lhs_nn, i);
            out = sqrt(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
        test_sqrt_real_axis(true, true);
    }

    void test_sqrt_pn()
    {
        std::transform(lhs_pn.cbegin(), lhs_pn.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::sqrt; return sqrt(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, lhs_pn, i);
            out = sqrt(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
        test_sqrt_real_axis(false, true);
    }

    void test_sqrt_np()
    {
        std::transform(lhs_np.cbegin(), lhs_np.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::sqrt; return sqrt(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, lhs_np, i);
            out = sqrt(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
        test_sqrt_real_axis(true, false);
    }

    void test_sqrt_pp()
    {
        std::transform(lhs_pp.cbegin(), lhs_pp.cend(), expected.begin(),
                       [](const value_type& v)
                       { using std::sqrt; return sqrt(v); });
        for (size_t i = 0; i < nb_input; i += size)
        {
            batch_type in, out, ref;
            detail::load_batch(in, lhs_pp, i);
            out = sqrt(in);
            detail::load_batch(ref, expected, i);
            CHECK_BATCH_EQ(ref, out);
        }
        test_sqrt_real_axis(false, false);
    }

    void test_sqrt_real_axis(bool negative_real, bool negative_imag)
    {
        const real_value_type zero(0);
        const std::array<real_value_type, 5> real_parts = {
            zero, std::numeric_limits<real_value_type>::denorm_min(),
            real_value_type(1), real_value_type(4), std::numeric_limits<real_value_type>::max()
        };
        std::array<value_type, size> inputs, results;
        std::array<real_value_type, size> magnitudes;

        for (size_t start = 0; start < real_parts.size(); start += size)
        {
            for (size_t lane = 0; lane < size; ++lane)
            {
                const auto x = real_parts[(start + lane) % real_parts.size()];
                const bool negative_y = negative_imag != ((start + lane) % 2 != 0);
                inputs[lane] = value_type(negative_real ? -x : x, negative_y ? -zero : zero);
            }

            const auto batch_inputs = batch_type::load_unaligned(inputs.data());
            xsimd::sqrt(batch_inputs).store_unaligned(results.data());
            // Preserve the real backend's rounding and underflow behavior.
            xsimd::sqrt(xsimd::abs(batch_inputs.real())).store_unaligned(magnitudes.data());

            for (size_t lane = 0; lane < size; ++lane)
            {
                const auto x = inputs[lane].real();
                const auto y = inputs[lane].imag();
                const auto magnitude = magnitudes[lane];
                const value_type ref = x < zero
                    ? value_type(zero, std::copysign(magnitude, y))
                    : value_type(magnitude, y);
                CHECK_EQ(results[lane].real(), ref.real());
                CHECK_EQ(results[lane].imag(), ref.imag());
#ifndef __FAST_MATH__
                // Fast-math does not preserve the side of the real-axis branch cut.
                CHECK_EQ(std::signbit(results[lane].real()), std::signbit(ref.real()));
                CHECK_EQ(std::signbit(results[lane].imag()), std::signbit(ref.imag()));
#endif
            }
        }
    }
};

TEST_CASE_TEMPLATE("[complex power]", B, BATCH_COMPLEX_TYPES)
{
    complex_power_test<B> Test;
    SUBCASE("abs")
    {
        Test.test_abs();
    }

    SUBCASE("arg")
    {
        Test.test_arg();
    }

    SUBCASE("pow")
    {
        Test.test_pow();
    }

    SUBCASE("pow real complex")
    {
        Test.test_pow_real_complex();
    }

    SUBCASE("pow complex real")
    {
        Test.test_pow_complex_real();
    }

    SUBCASE("sqrt_nn")
    {
        Test.test_sqrt_nn();
    }

    SUBCASE("sqrt_pn")
    {
        Test.test_sqrt_pn();
    }

    SUBCASE("sqrt_np")
    {
        Test.test_sqrt_np();
    }

    SUBCASE("sqrt_pp")
    {
        Test.test_sqrt_pp();
    }
}
#endif
