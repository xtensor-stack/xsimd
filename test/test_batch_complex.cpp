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

#include <cmath>
#include <functional>
#include <numeric>

#include "test_utils.hpp"

using namespace std::placeholders;

template <class B>
struct batch_complex_test
{
    using batch_type = xsimd::simd_type<typename B::value_type>;
    using arch_type = typename B::arch_type;
    using real_batch_type = typename B::real_batch;
    using value_type = typename B::value_type;
    using real_value_type = typename value_type::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using bool_array_type = std::array<bool, size>;
    using real_array_type = std::array<real_value_type, size>;

    array_type lhs;
    array_type rhs;
    value_type scalar;
    real_value_type real_scalar;

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    using xtl_value_type = xtl::xcomplex<real_value_type, real_value_type, true>;
    using xtl_array_type = std::array<xtl_value_type, size>;
#endif

    batch_complex_test()
    {
        scalar = value_type(real_value_type(1.4), real_value_type(2.3));
        real_scalar = scalar.real();
        for (size_t i = 0; i < size; ++i)
        {
            lhs[i] = value_type(real_value_type(i) / real_value_type(4) + real_value_type(1.2) * std::sqrt(real_value_type(i + 0.25)),
                                real_value_type(i) / real_value_type(5));
            rhs[i] = value_type(real_value_type(10.2) / real_value_type(i + 2) + real_value_type(0.25), real_value_type(i) / real_value_type(3.2));
        }
    }

    void test_load_store() const
    {
        {
            array_type res;
            batch_type b = batch_type::load_unaligned(lhs.data());
            b.store_unaligned(res.data());
            CHECK_EQ(res, lhs);

            alignas(arch_type::alignment()) array_type arhs(this->rhs);
            alignas(arch_type::alignment()) array_type ares;
            b = batch_type::load_aligned(arhs.data());
            b.store_aligned(ares.data());
            CHECK_EQ(ares, rhs);
        }

        {
            real_array_type real, imag, res_real, res_imag;
            for (size_t i = 0; i < size; ++i)
            {
                real[i] = lhs[i].real();
                imag[i] = lhs[i].imag();
            }
            batch_type b = batch_type::load_unaligned(real.data(), imag.data());
            b.store_unaligned(res_real.data(), res_imag.data());
            CHECK_EQ(res_real, real);

            alignas(arch_type::alignment()) real_array_type areal, aimag, ares_real, ares_imag;
            for (size_t i = 0; i < size; ++i)
            {
                areal[i] = lhs[i].real();
                aimag[i] = lhs[i].imag();
            }
            b = batch_type::load_aligned(areal.data(), aimag.data());
            b.store_aligned(ares_real.data(), ares_imag.data());
            CHECK_EQ(ares_real, areal);
        }
        {
            real_array_type real, imag, res_real, res_imag;
            for (size_t i = 0; i < size; ++i)
            {
                real[i] = lhs[i].real();
                imag[i] = 0;
            }
            batch_type b = batch_type::load_unaligned(real.data());
            b.store_unaligned(res_real.data(), res_imag.data());
            CHECK_EQ(res_real, real);
            CHECK_EQ(res_imag, imag);

            alignas(arch_type::alignment()) real_array_type areal, aimag, ares_real, ares_imag;
            for (size_t i = 0; i < size; ++i)
            {
                areal[i] = lhs[i].real();
                aimag[i] = 0;
            }
            b = batch_type::load_aligned(areal.data());
            b.store_aligned(ares_real.data(), ares_imag.data());
            CHECK_EQ(ares_real, areal);
            CHECK_EQ(ares_imag, aimag);
        }
    }
#ifdef XSIMD_ENABLE_XTL_COMPLEX
    void test_load_store_xtl() const
    {
        xtl_array_type tmp;
        std::fill(tmp.begin(), tmp.end(), xtl_value_type(2, 3));

        alignas(arch_type::alignment()) xtl_array_type aligned_tmp;
        std::fill(aligned_tmp.begin(), aligned_tmp.end(), xtl_value_type(2, 3));

        batch_type b0(xtl_value_type(2, 3));
        CHECK_EQ(b0, tmp);

        batch_type b1 = xsimd::load_as<xtl_value_type>(aligned_tmp.data(), xsimd::aligned_mode());
        CHECK_EQ(b1, tmp);

        batch_type b2 = xsimd::load_as<xtl_value_type>(tmp.data(), xsimd::unaligned_mode());
        CHECK_EQ(b2, tmp);

        xsimd::store_as(aligned_tmp.data(), b1, xsimd::aligned_mode());
        CHECK_EQ(b1, aligned_tmp);

        xsimd::store_as(tmp.data(), b2, xsimd::unaligned_mode());
        CHECK_EQ(b2, tmp);
    }
#endif

    void test_constructors() const
    {
        array_type tmp;
        std::fill(tmp.begin(), tmp.end(), value_type(2, 3));
        batch_type b0a(value_type(2, 3));
        CHECK_EQ(b0a, tmp);

        batch_type b0b = batch_type::broadcast(value_type(2, 3));
        CHECK_EQ(b0b, tmp);

        batch_type b0c = xsimd::broadcast(value_type(2, 3));
        CHECK_EQ(b0c, tmp);

        std::fill(tmp.begin(), tmp.end(), value_type(real_scalar));
        batch_type b1(real_scalar);
        CHECK_EQ(b1, tmp);

        real_array_type real, imag;
        for (size_t i = 0; i < size; ++i)
        {
            real[i] = lhs[i].real();
            imag[i] = lhs[i].imag();
            tmp[i] = value_type(real[i]);
        }
    }

    void test_access_operator() const
    {
        batch_type res = batch_lhs();
        for (size_t i = 0; i < size; ++i)
        {
            CHECK_EQ(res.get(i), lhs[i]);
        }
    }

    void test_arithmetic() const
    {
        // +batch
        {
            array_type expected = lhs;
            batch_type res = +batch_lhs();
            CHECK_BATCH_EQ(res, expected);
        }
        // -batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::negate<value_type>());
            batch_type res = -batch_lhs();
            CHECK_BATCH_EQ(res, expected);
        }
        // batch + batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::plus<value_type>());
            batch_type res = batch_lhs() + batch_rhs();
            CHECK_BATCH_EQ(res, expected);
        }
        // batch + scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::plus<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() + scalar;
            CHECK_BATCH_EQ(lres, expected);
            batch_type rres = scalar + batch_lhs();
            CHECK_BATCH_EQ(rres, expected);
        }

        // batch + real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l + r.real(); });
            batch_type lres = batch_lhs() + batch_rhs().real();
            CHECK_BATCH_EQ(lres, expected);
            batch_type rres = batch_rhs().real() + batch_lhs();
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch + real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::plus<value_type>(), _1, real_scalar));
            batch_type lres = batch_lhs() + real_scalar;
            CHECK_BATCH_EQ(lres, expected);
            batch_type rres = real_scalar + batch_lhs();
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch - batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::minus<value_type>());
            batch_type res = batch_lhs() - batch_rhs();
            CHECK_BATCH_EQ(res, expected);
        }
        // batch - scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() - scalar;
            CHECK_BATCH_EQ(lres, expected);
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), scalar, _1));
            batch_type rres = scalar - batch_lhs();
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch - real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l - r.real(); });
            batch_type lres = batch_lhs() - batch_rhs().real();
            CHECK_BATCH_EQ(lres, expected);
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return r.real() - l; });
            batch_type rres = batch_rhs().real() - batch_lhs();
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch - real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), _1, real_scalar));
            batch_type lres = batch_lhs() - real_scalar;
            CHECK_BATCH_EQ(lres, expected);
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), real_scalar, _1));
            batch_type rres = real_scalar - batch_lhs();
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch * batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::multiplies<value_type>());
            batch_type res = batch_lhs() * batch_rhs();
            CHECK_BATCH_EQ(res, expected);
        }
        // batch * scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::multiplies<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() * scalar;
            CHECK_BATCH_EQ(lres, expected);
            batch_type rres = scalar * batch_lhs();
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch * real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l * r.real(); });
            batch_type lres = batch_lhs() * batch_rhs().real();
            CHECK_BATCH_EQ(lres, expected);
            batch_type rres = batch_rhs().real() * batch_lhs();
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch * real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::multiplies<value_type>(), _1, real_scalar));
            batch_type lres = batch_lhs() * real_scalar;
            CHECK_BATCH_EQ(lres, expected);
            batch_type rres = real_scalar * batch_lhs();
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch / batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::divides<value_type>());
            batch_type res = batch_lhs() / batch_rhs();
            CHECK_BATCH_EQ(res, expected);
        }
        // batch / scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() / scalar;
            CHECK_BATCH_EQ(lres, expected);
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), scalar, _1));
            batch_type rres = scalar / batch_lhs();
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch / real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l / r.real(); });
            batch_type lres = batch_lhs() / batch_rhs().real();
            CHECK_BATCH_EQ(lres, expected);
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return r.real() / l; });
            batch_type rres = batch_rhs().real() / batch_lhs();
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch - real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), _1, real_scalar));
            batch_type lres = batch_lhs() / real_scalar;
            CHECK_BATCH_EQ(lres, expected);
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), real_scalar, _1));
            batch_type rres = real_scalar / batch_lhs();
            CHECK_BATCH_EQ(rres, expected);
        }
    }

    void test_computed_assignment() const
    {

        // batch += batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::plus<value_type>());
            batch_type res = batch_lhs();
            res += batch_rhs();
            CHECK_BATCH_EQ(res, expected);
        }
        // batch += scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::plus<value_type>(), _1, scalar));
            batch_type res = batch_lhs();
            res += scalar;
            CHECK_BATCH_EQ(res, expected);
        }
        // batch += real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l + r.real(); });
            batch_type res = batch_lhs();
            res += batch_rhs().real();
            CHECK_BATCH_EQ(res, expected);
        }
        // batch += real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::plus<value_type>(), _1, real_scalar));
            batch_type res = batch_lhs();
            res += real_scalar;
            CHECK_BATCH_EQ(res, expected);
        }
        // batch -= batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::minus<value_type>());
            batch_type res = batch_lhs();
            res -= batch_rhs();
            CHECK_BATCH_EQ(res, expected);
        }
        // batch -= scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), _1, scalar));
            batch_type res = batch_lhs();
            res -= scalar;
            CHECK_BATCH_EQ(res, expected);
        }
        // batch -= real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l - r.real(); });
            batch_type res = batch_lhs();
            res -= batch_rhs().real();
            CHECK_BATCH_EQ(res, expected);
        }
        // batch -= real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), _1, real_scalar));
            batch_type res = batch_lhs();
            res -= real_scalar;
            CHECK_BATCH_EQ(res, expected);
        }
        // batch *= batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::multiplies<value_type>());
            batch_type res = batch_lhs();
            res *= batch_rhs();
            CHECK_BATCH_EQ(res, expected);
        }
        // batch *= scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::multiplies<value_type>(), _1, scalar));
            batch_type res = batch_lhs();
            res *= scalar;
            CHECK_BATCH_EQ(res, expected);
        }
        // batch *= real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l * r.real(); });
            batch_type res = batch_lhs();
            res *= batch_rhs().real();
            CHECK_BATCH_EQ(res, expected);
        }
        // batch *= real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::multiplies<value_type>(), _1, real_scalar));
            batch_type res = batch_lhs();
            res *= real_scalar;
            CHECK_BATCH_EQ(res, expected);
        }
        // batch /= batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::divides<value_type>());
            batch_type res = batch_lhs();
            res /= batch_rhs();
            CHECK_BATCH_EQ(res, expected);
        }
        // batch /= scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), _1, scalar));
            batch_type res = batch_lhs();
            res /= scalar;
            CHECK_BATCH_EQ(res, expected);
        }
        // batch /= real_batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l / r.real(); });
            batch_type res = batch_lhs();
            res /= batch_rhs().real();
            CHECK_BATCH_EQ(res, expected);
        }
        // batch /= real_scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), _1, real_scalar));
            batch_type res = batch_lhs();
            res /= real_scalar;
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_conj_norm_proj() const
    {
        // conj
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& v)
                           { using std::conj; return conj(v); });
            batch_type res = conj(batch_lhs());
            CHECK_BATCH_EQ(res, expected);
        }
        // norm
        {
            real_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& v)
                           { using std::norm; return norm(v); });
            real_batch_type res = norm(batch_lhs());
            CHECK_BATCH_EQ(res, expected);
        }
        // proj
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& v)
                           { using std::proj; return proj(v); });
            batch_type res = proj(batch_lhs());
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_conj_norm_proj_real() const
    {
        // conj real batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::conj(std::real(v)); });
            batch_type res = conj(real(batch_lhs()));
            CHECK_BATCH_EQ(res, expected);
        }
        // norm real batch
        {
            real_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::norm(std::real(v)); });
            real_batch_type res = norm(real(batch_lhs()));
            CHECK_BATCH_EQ(res, expected);
        }
        // proj real batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& v)
                           { return std::proj(std::real(v)); });
            batch_type res = proj(real(batch_lhs()));
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_polar() const
    {
        // polar w/ magnitude/phase
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), expected.begin(),
                           [](const value_type& v_lhs, const value_type& v_rhs)
                           { return std::polar(std::real(v_lhs), std::real(v_rhs)); });
            batch_type res = polar(real(batch_lhs()), real(batch_rhs()));
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_horizontal_operations() const
    {
        // reduce_add
        {
            value_type expected = std::accumulate(lhs.cbegin(), lhs.cend(), value_type(0));
            value_type res = reduce_add(batch_lhs());
            CHECK_SCALAR_EQ(res, expected);
        }
    }

    void test_fused_operations() const
    {
        // fma
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l * r + r; });
            batch_type res = xsimd::fma(batch_lhs(), batch_rhs(), batch_rhs());
            CHECK_BATCH_EQ(res, expected);
        }
        // fms
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l * r - r; });
            batch_type res = fms(batch_lhs(), batch_rhs(), batch_rhs());
            CHECK_BATCH_EQ(res, expected);
        }

        // fnma
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return -l * r + r; });
            batch_type res = fnma(batch_lhs(), batch_rhs(), batch_rhs());
            CHECK_BATCH_EQ(res, expected);
        }
        // fnms
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return -l * r - r; });
            batch_type res = fnms(batch_lhs(), batch_rhs(), batch_rhs());
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_boolean_conversion() const
    {
        // !batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& l)
                           { return l == value_type(0); });
            batch_type res = (batch_type)!batch_lhs();
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_isnan() const
    {
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& l)
                           { return std::isnan(l.real()) || std::isnan(l.imag()); });
            typename batch_type::batch_bool_type res = isnan(batch_lhs());
            CHECK_BATCH_EQ(res, expected);
        }
    }

private:
    batch_type batch_lhs() const
    {
        batch_type res = batch_type::load_unaligned(lhs.data());
        return res;
    }

    batch_type batch_rhs() const
    {
        batch_type res = batch_type::load_unaligned(rhs.data());
        return res;
    }
};

TEST_CASE_TEMPLATE("[xsimd complex batches]", B, BATCH_COMPLEX_TYPES)
{
    batch_complex_test<B> Test;
    SUBCASE("load_store") { Test.test_load_store(); }

#ifdef XSIMD_ENABLE_XTL_COMPLEX
    SUBCASE("load_store_xtl")
    {
        Test.test_load_store_xtl();
    }
#endif

    SUBCASE("constructors")
    {
        Test.test_constructors();
    }

    SUBCASE("access_operator") { Test.test_access_operator(); }

    SUBCASE("arithmetic") { Test.test_arithmetic(); }

    SUBCASE("computed_assignment") { Test.test_computed_assignment(); }

    SUBCASE("conj_norm_proj") { Test.test_conj_norm_proj(); }

    SUBCASE("conj_norm_proj_real") { Test.test_conj_norm_proj_real(); }

    SUBCASE("polar") { Test.test_polar(); }

    SUBCASE("horizontal_operations") { Test.test_horizontal_operations(); }

    SUBCASE("fused_operations") { Test.test_fused_operations(); }

    SUBCASE("boolean_conversion") { Test.test_boolean_conversion(); }

    SUBCASE("isnan") { Test.test_isnan(); }
}
#endif
