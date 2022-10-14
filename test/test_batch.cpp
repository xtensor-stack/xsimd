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
#include <sstream>

#include "test_utils.hpp"

using namespace std::placeholders;

template <class B>
struct batch_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using bool_array_type = std::array<bool, size>;

    array_type lhs;
    array_type rhs;
    value_type scalar;

    batch_test()
    {
        init_operands();
    }

    void test_stream_dump() const
    {
        array_type res;
        batch_type b = batch_type::load_unaligned(lhs.data());
        b.store_unaligned(res.data());

        std::ostringstream b_dump;
        b_dump << b;

        std::ostringstream res_dump;
        res_dump << '(';
        for (std::size_t i = 0; i < res.size() - 1; ++i)
            res_dump << res[i] << ", ";
        res_dump << res.back() << ')';

        CHECK_EQ(res_dump.str(), b_dump.str());
    }

    void test_load_store() const
    {
        array_type res;
        batch_type b = batch_type::load_unaligned(lhs.data());
        b.store_unaligned(res.data());
        INFO("load_unaligned / store_unaligned");
        CHECK_EQ(res, lhs);

        alignas(xsimd::default_arch::alignment()) array_type arhs(this->rhs);
        alignas(xsimd::default_arch::alignment()) array_type ares;
        b = batch_type::load_aligned(arhs.data());
        b.store_aligned(ares.data());
        INFO("load_aligned / store_aligned");
        CHECK_EQ(ares, rhs);
    }

    template <size_t... Is>
    struct pack
    {
    };

    template <size_t... Values>
    void check_constructor_from_sequence(std::integral_constant<size_t, 0>, pack<Values...>) const
    {
        array_type tmp = { static_cast<value_type>(Values)... };
        batch_type b0(static_cast<value_type>(Values)...);
        INFO("batch(values...)");
        CHECK_EQ(b0, tmp);

        batch_type b1 { static_cast<value_type>(Values)... };
        INFO("batch{values...}");
        CHECK_EQ(b0, tmp);
    }

    template <size_t I, size_t... Values>
    void check_constructor_from_sequence(std::integral_constant<size_t, I>, pack<Values...>) const
    {
        return check_constructor_from_sequence(std::integral_constant<size_t, I - 1>(), pack<Values..., I>());
    }

    void test_constructors() const
    {
        batch_type b;
        // value initialized to random data, can't be checked

        array_type tmp;
        std::fill(tmp.begin(), tmp.end(), value_type(2));
        batch_type b0a(2);
        INFO("batch(value_type)");
        CHECK_EQ(b0a, tmp);

        batch_type b0b { 2 };
        INFO("batch{value_type}");
        CHECK_EQ(b0b, tmp);

        check_constructor_from_sequence(std::integral_constant<size_t, size>(), pack<>());
    }

    void test_static_builders() const
    {
        {
            array_type expected;
            std::fill(expected.begin(), expected.end(), value_type(2));

            auto res = batch_type::broadcast(value_type(2));
            INFO("batch::broadcast");
            CHECK_EQ(res, expected);
        }
        {
            array_type res;
            auto b = batch_type::load_unaligned(lhs.data());
            b.store_unaligned(res.data());
            INFO("batch::load_unaligned");
            CHECK_EQ(res, lhs);
        }
        {
            alignas(xsimd::default_arch::alignment()) array_type arhs(this->rhs);
            alignas(xsimd::default_arch::alignment()) array_type ares;
            auto b = batch_type::load_aligned(arhs.data());
            b.store_aligned(ares.data());
            INFO("batch::load_aligned");
            CHECK_EQ(ares, rhs);
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
            INFO("+batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // -batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::negate<value_type>());
            batch_type res = -batch_lhs();
            INFO("-batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch + batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::plus<value_type>());
            batch_type res = batch_lhs() + batch_rhs();
            INFO("batch + batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch + scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::plus<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() + scalar;
            INFO("batch + scalar");
            CHECK_BATCH_EQ(lres, expected);
            batch_type rres = scalar + batch_lhs();
            INFO("scalar + batch");
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch - batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::minus<value_type>());
            batch_type res = batch_lhs() - batch_rhs();
            INFO("batch - batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch - scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() - scalar;
            INFO("batch - scalar");
            CHECK_BATCH_EQ(lres, expected);
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), scalar, _1));
            batch_type rres = scalar - batch_lhs();
            INFO("scalar - batch");
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch * batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::multiplies<value_type>());
            batch_type res = batch_lhs() * batch_rhs();
            INFO("batch * batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch * scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::multiplies<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() * scalar;
            INFO("batch * scalar");
            CHECK_BATCH_EQ(lres, expected);
            batch_type rres = scalar * batch_lhs();
            INFO("scalar * batch");
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch / batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::divides<value_type>());
            batch_type res = batch_lhs() / batch_rhs();
            INFO("batch / batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch / scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() / scalar;
            INFO("batch / scalar");
            CHECK_BATCH_EQ(lres, expected);
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), scalar, _1));
            batch_type rres = scalar / batch_lhs();
            INFO("scalar / batch");
            CHECK_BATCH_EQ(rres, expected);
        }
    }

    void test_saturated_arithmetic() const
    {
#ifdef T
        // batch + batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), xsimd::sadd<value_type>);
            batch_type res = xsimd::sadd(batch_lhs(), batch_rhs());
            INFO("sadd(batch, batch)");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch + scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), [this](value_type x)
                           { return xsimd::sadd(x, scalar); });
            batch_type lres = xsimd::sadd(batch_lhs(), scalar);
            INFO("sadd(batch, scalar)");
            CHECK_BATCH_EQ(lres, expected);
            batch_type rres = xsimd::sadd(scalar, batch_lhs());
            INFO("sadd(scalar, batch)");
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch - batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), [](value_type x, value_type y)
                           { return xsimd::ssub(x, y); });
            batch_type res = xsimd::ssub(batch_lhs(), batch_rhs());
            INFO("ssub(batch, batch)");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch - scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), [this](value_type x)
                           { return xsimd::ssub(x, scalar); });
            batch_type lres = xsimd::ssub(batch_lhs(), scalar);
            INFO("ssub(batch, scalar)");
            CHECK_BATCH_EQ(lres, expected);
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), [this](value_type x)
                           { return xsimd::ssub(scalar, x); });
            batch_type rres = xsimd::ssub(scalar, batch_lhs());
            INFO("ssub(scalar, batch)");
            CHECK_BATCH_EQ(rres, expected);
        }
#endif
    }

    void test_computed_assignment() const
    {
        // batch += batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::plus<value_type>());
            batch_type res = batch_lhs();
            res += batch_rhs();
            INFO("batch += batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch += scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::plus<value_type>(), _1, scalar));
            batch_type res = batch_lhs();
            res += scalar;
            INFO("batch += scalar");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch -= batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::minus<value_type>());
            batch_type res = batch_lhs();
            res -= batch_rhs();
            INFO("batch -= batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch -= scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::minus<value_type>(), _1, scalar));
            batch_type res = batch_lhs();
            res -= scalar;
            INFO("batch -= scalar");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch *= batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::multiplies<value_type>());
            batch_type res = batch_lhs();
            res *= batch_rhs();
            INFO("batch *= batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch *= scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::multiplies<value_type>(), _1, scalar));
            batch_type res = batch_lhs();
            res *= scalar;
            INFO("batch *= scalar");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch /= batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::divides<value_type>());
            batch_type res = batch_lhs();
            res /= batch_rhs();
            INFO("batch /= batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch /= scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::divides<value_type>(), _1, scalar));
            batch_type res = batch_lhs();
            res /= scalar;
            INFO("batch /= scalar");
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_comparison() const
    {

        // batch == batch
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l == r; });
            auto res = batch_lhs() == batch_rhs();
            INFO("batch == batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch == scalar
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [this](const value_type& l)
                           { return l == scalar; });
            auto res = batch_lhs() == scalar;
            INFO("batch == scalar");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch != batch
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l != r; });
            auto res = batch_lhs() != batch_rhs();
            INFO("batch != batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch != scalar
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [this](const value_type& l)
                           { return l != scalar; });
            auto res = batch_lhs() != scalar;
            INFO("batch != scalar");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch < batch
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l < r; });
            auto res = batch_lhs() < batch_rhs();
            INFO("batch < batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch < scalar
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [this](const value_type& l)
                           { return l < scalar; });
            auto res = batch_lhs() < scalar;
            INFO("batch < scalar");
            CHECK_BATCH_EQ(res, expected);
        }

        // batch <= batch
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l <= r; });
            auto res = batch_lhs() <= batch_rhs();
            INFO("batch <= batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch <= scalar
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [this](const value_type& l)
                           { return l <= scalar; });
            auto res = batch_lhs() <= scalar;
            INFO("batch <= scalar");
            CHECK_BATCH_EQ(res, expected);
        }

        // batch > batch
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l > r; });
            auto res = batch_lhs() > batch_rhs();
            INFO("batch > batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch > scalar
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [this](const value_type& l)
                           { return l > scalar; });
            auto res = batch_lhs() > scalar;
            INFO("batch > scalar");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch >= batch
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l >= r; });
            auto res = batch_lhs() >= batch_rhs();
            INFO("batch >= batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch >= scalar
        {
            bool_array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [this](const value_type& l)
                           { return l >= scalar; });
            auto res = batch_lhs() >= scalar;
            INFO("batch >= scalar");
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_logical() const
    {
        // batch && batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::logical_and<value_type>());
            batch_type res = batch_lhs() && batch_rhs();
            INFO("batch && batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch && scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::logical_and<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() && scalar;
            INFO("batch && scalar");
            CHECK_BATCH_EQ(lres, expected);
            batch_type rres = scalar && batch_lhs();
            INFO("scalar && batch");
            CHECK_BATCH_EQ(rres, expected);
        }
        // batch || batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), std::logical_or<value_type>());
            batch_type res = batch_lhs() || batch_rhs();
            INFO("batch && batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch || scalar
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), std::bind(std::logical_or<value_type>(), _1, scalar));
            batch_type lres = batch_lhs() || scalar;
            INFO("batch || scalar");
            CHECK_BATCH_EQ(lres, expected);
            batch_type rres = scalar || batch_lhs();
            INFO("scalar || batch");
            CHECK_BATCH_EQ(rres, expected);
        }
    }

    void test_min_max() const
    {
        // min
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return std::min(l, r); });
            batch_type res = min(batch_lhs(), batch_rhs());
            INFO("min");
            CHECK_BATCH_EQ(res, expected);
        }
        // min limit case
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type&, const value_type& r)
                           { return std::min(std::numeric_limits<value_type>::min(), r); });
            batch_type res = xsimd::min(batch_type(std::numeric_limits<value_type>::min()), batch_rhs());
            INFO("min limit");
            CHECK_BATCH_EQ(res, expected);
        }
        // fmin
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return std::fmin(l, r); });
            batch_type res = min(batch_lhs(), batch_rhs());
            INFO("fmin");
            CHECK_BATCH_EQ(res, expected);
        }
        // max
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return std::max(l, r); });
            batch_type res = max(batch_lhs(), batch_rhs());
            INFO("max");
            CHECK_BATCH_EQ(res, expected);
        }
        // max limit case
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type&, const value_type& r)
                           { return std::max(std::numeric_limits<value_type>::max(), r); });
            batch_type res = xsimd::max(batch_type(std::numeric_limits<value_type>::max()), batch_rhs());
            INFO("max limit");
            CHECK_BATCH_EQ(res, expected);
        }
        // fmax
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return std::fmax(l, r); });
            batch_type res = fmax(batch_lhs(), batch_rhs());
            INFO("fmax");
            CHECK_BATCH_EQ(res, expected);
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
            // Warning: ADL seems to not work correctly on Windows, thus the full qualified call
            batch_type res = xsimd::fma(batch_lhs(), batch_rhs(), batch_rhs());
            INFO("fma");
            CHECK_BATCH_EQ(res, expected);
        }
        // fms
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return l * r - r; });
            batch_type res = fms(batch_lhs(), batch_rhs(), batch_rhs());
            INFO("fms");
            CHECK_BATCH_EQ(res, expected);
        }
        // fnma
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return -l * r + r; });
            batch_type res = fnma(batch_lhs(), batch_rhs(), batch_rhs());
            INFO("fnma");
            CHECK_BATCH_EQ(res, expected);
        }
        // fnms
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), expected.begin(),
                           [](const value_type& l, const value_type& r)
                           { return -l * r - r; });
            batch_type res = fnms(batch_lhs(), batch_rhs(), batch_rhs());
            INFO("fnms");
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_abs() const
    {
        // abs
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& l)
                           { return ::detail::uabs(l); });
            batch_type res = abs(batch_lhs());
            INFO("abs");
            CHECK_BATCH_EQ(res, expected);
        }
        // fabs
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& l)
                           { return std::fabs(l); });
            batch_type res = fabs(batch_lhs());
            INFO("fabs");
            CHECK_BATCH_EQ(res, expected);
        }
    }

    void test_horizontal_operations() const
    {
        // reduce_add
        {
            value_type expected = std::accumulate(lhs.cbegin(), lhs.cend(), value_type(0));
            value_type res = reduce_add(batch_lhs());
            INFO("reduce_add");
            CHECK_SCALAR_EQ(res, expected);
        }
        // reduce_max
        {
            value_type expected = *std::max_element(lhs.cbegin(), lhs.cend());
            value_type res = reduce_max(batch_lhs());
            INFO("reduce_max");
            CHECK_SCALAR_EQ(res, expected);
        }
        // reduce_min
        {
            value_type expected = *std::min_element(lhs.cbegin(), lhs.cend());
            value_type res = reduce_min(batch_lhs());
            INFO("reduce_min");
            CHECK_SCALAR_EQ(res, expected);
        }
    }

    void test_boolean_conversions() const
    {
        using batch_bool_type = typename batch_type::batch_bool_type;
        // batch = true
        {
            batch_bool_type tbt(true);
            batch_type expected = batch_type(value_type(1));
            batch_type res = (batch_type)tbt;
            INFO("batch = true");
            CHECK_BATCH_EQ(res, expected);
        }
        // batch = false
        {
            batch_bool_type fbt(false);
            batch_type expected = batch_type(value_type(0));
            batch_type res = (batch_type)fbt;
            INFO("batch = false");
            CHECK_BATCH_EQ(res, expected);
        }
        // !batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](const value_type& l)
                           { return !l; });
            batch_type res = (batch_type)!batch_lhs();
            INFO("!batch");
            CHECK_BATCH_EQ(res, expected);
        }
        // bitwise_cast
        {
            batch_bool_type fbt(false);
            batch_type expected = batch_type(value_type(0));
            batch_type res = bitwise_cast(fbt);
            INFO("bitwise_cast");
            CHECK_BATCH_EQ(res, expected);
        }
        // bitwise not
        {
            batch_bool_type fbt(true);
            batch_type expected = batch_type(value_type(0));
            batch_type res = ~bitwise_cast(fbt);
            INFO("~batch");
            CHECK_BATCH_EQ(res, expected);
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

    template <class T = value_type>
    xsimd::enable_integral_t<T, void> init_operands()
    {
        for (size_t i = 0; i < size; ++i)
        {
            bool negative_lhs = std::is_signed<T>::value && (i % 2 == 1);
            lhs[i] = value_type(i) * (negative_lhs ? -10 : 10);
            if (lhs[i] == value_type(0))
            {
                lhs[i] += value_type(1);
            }
            rhs[i] = value_type(i) + value_type(4);
        }
        scalar = value_type(3);
    }

    template <class T = value_type>
    xsimd::enable_floating_point_t<T, void> init_operands()
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
        scalar = value_type(1.2);
    }
};

TEST_CASE_TEMPLATE("[batch]", B, BATCH_TYPES)
{
    batch_test<B> Test;

    SUBCASE("stream_dump")
    {
        Test.test_stream_dump();
    }

    SUBCASE("load_store")
    {
        Test.test_load_store();
    }

    SUBCASE("constructors")
    {
        Test.test_constructors();
    }

    SUBCASE("static_builders")
    {
        Test.test_static_builders();
    }

    SUBCASE("access_operator")
    {
        Test.test_access_operator();
    }

    SUBCASE("arithmetic")
    {
        Test.test_arithmetic();
    }

    SUBCASE("saturated_arithmetic")
    {
        Test.test_saturated_arithmetic();
    }

    SUBCASE("computed_assignment")
    {
        Test.test_computed_assignment();
    }

    SUBCASE("comparison")
    {
        Test.test_comparison();
    }
    SUBCASE("logical")
    {
        Test.test_logical();
    }

    SUBCASE("min_max")
    {
        Test.test_min_max();
    }

    SUBCASE("fused_operations")
    {
        Test.test_fused_operations();
    }

    SUBCASE("abs")
    {
        Test.test_abs();
    }

    SUBCASE("horizontal_operations")
    {
        Test.test_horizontal_operations();
    }

    SUBCASE("boolean_conversions")
    {
        Test.test_boolean_conversions();
    }
}
#endif
