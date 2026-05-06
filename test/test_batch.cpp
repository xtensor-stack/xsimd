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

namespace detail_test_mulhilo
{
    template <class T>
    typename std::enable_if<std::is_integral<T>::value && (sizeof(T) <= 4), T>::type
    mulhi_reference(T x, T y) noexcept
    {
        using W = typename std::conditional<std::is_signed<T>::value, int64_t, uint64_t>::type;
        return static_cast<T>((static_cast<W>(x) * static_cast<W>(y)) >> (8 * sizeof(T)));
    }

#if defined(__SIZEOF_INT128__)
    template <class T>
    typename std::enable_if<std::is_integral<T>::value && (sizeof(T) == 8), T>::type
    mulhi_reference(T x, T y) noexcept
    {
        using W = typename std::conditional<std::is_signed<T>::value, __int128, unsigned __int128>::type;
        return static_cast<T>((static_cast<W>(x) * static_cast<W>(y)) >> 64);
    }
#else
    template <class T>
    typename std::enable_if<std::is_integral<T>::value && (sizeof(T) == 8), T>::type
    mulhi_reference(T x, T y) noexcept
    {
        uint64_t ux = static_cast<uint64_t>(x);
        uint64_t uy = static_cast<uint64_t>(y);
        uint64_t xl = ux & 0xffffffffULL, xh = ux >> 32;
        uint64_t yl = uy & 0xffffffffULL, yh = uy >> 32;
        uint64_t ll = xl * yl, lh = xl * yh, hl = xh * yl, hh = xh * yh;
        uint64_t mid = (ll >> 32) + (lh & 0xffffffffULL) + (hl & 0xffffffffULL);
        uint64_t hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
        if (std::is_signed<T>::value)
        {
            if (x < 0)
                hi -= uy;
            if (y < 0)
                hi -= ux;
        }
        return static_cast<T>(hi);
    }
#endif
}
using detail_test_mulhilo::mulhi_reference;

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
        (void)b;

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

    void test_first_element() const
    {
        batch_type res = batch_lhs();
        CHECK_EQ(res.first(), lhs[0]);
    }

    template <size_t... Is>
    void test_get_impl(batch_type const& res, std::index_sequence<Is...>) const
    {
        array_type extracted = { xsimd::get<Is>(res)... };
        CHECK_EQ(extracted, lhs);
        CHECK_BATCH_EQ(batch_type::load_unaligned(extracted.data()), res);
    }

    void test_get() const
    {
        batch_type res = batch_lhs();
        CHECK_EQ(xsimd::get<0>(res), res.first());
        test_get_impl(res, std::make_index_sequence<size> {});
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

    void test_incr_decr() const
    {
        // incr
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), xsimd::incr<value_type>);
            batch_type res = xsimd::incr(batch_lhs());
            INFO("incr(batch)");
            CHECK_BATCH_EQ(res, expected);
        }

        // incr_if
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](value_type v)
                           { return v > 1 ? v + 1 : v; });
            batch_type res = xsimd::incr_if(batch_lhs(), batch_lhs() > value_type(1));
            INFO("incr_if(batch)");
            CHECK_BATCH_EQ(res, expected);
        }

        // decr
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(), xsimd::decr<value_type>);
            batch_type res = xsimd::decr(batch_lhs());
            INFO("decr(batch)");
            CHECK_BATCH_EQ(res, expected);
        }

        // decr_if
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), expected.begin(),
                           [](value_type v)
                           { return v > 1 ? v - 1 : v; });
            batch_type res = xsimd::decr_if(batch_lhs(), batch_lhs() > value_type(1));
            INFO("decr_if(batch)");
            CHECK_BATCH_EQ(res, expected);
        }
    }

    template <class U = value_type>
    void test_mulhilo_impl(std::true_type /*integral*/) const
    {
        using UT = typename std::make_unsigned<value_type>::type;

        auto run_case = [](array_type const& a, array_type const& b, const char* tag)
        {
            batch_type ba = batch_type::load_unaligned(a.data());
            batch_type bb = batch_type::load_unaligned(b.data());

            array_type lo_expected;
            array_type hi_expected;
            for (std::size_t i = 0; i < size; ++i)
            {
                lo_expected[i] = static_cast<value_type>(static_cast<UT>(a[i]) * static_cast<UT>(b[i]));
                hi_expected[i] = mulhi_reference(a[i], b[i]);
            }

            batch_type lo_res = xsimd::mullo(ba, bb);
            INFO("mullo(batch, batch) [" << tag << "]");
            CHECK_BATCH_EQ(lo_res, lo_expected);

            batch_type hi_res = xsimd::mulhi(ba, bb);
            INFO("mulhi(batch, batch) [" << tag << "]");
            CHECK_BATCH_EQ(hi_res, hi_expected);

            auto p = xsimd::mulhilo(ba, bb);
            INFO("mulhilo.first == mulhi [" << tag << "]");
            CHECK_BATCH_EQ(p.first, hi_res);
            INFO("mulhilo.second == mullo [" << tag << "]");
            CHECK_BATCH_EQ(p.second, lo_res);
        };

        // baseline: small operands from init_operands
        run_case(lhs, rhs, "small");

        // edge operands that actually exercise the high half
        constexpr value_type vmin = std::numeric_limits<value_type>::min();
        constexpr value_type vmax = std::numeric_limits<value_type>::max();
        constexpr bool is_signed = std::is_signed<value_type>::value;

        // Pattern A: extremes paired with extremes (covers vmax*vmax, vmin*vmin,
        // vmin*vmax, vmin*-1 — the classic signed-overflow corners).
        {
            array_type a, b;
            for (std::size_t i = 0; i < size; ++i)
            {
                switch (i % 8)
                {
                case 0:
                    a[i] = vmax;
                    b[i] = vmax;
                    break;
                case 1:
                    a[i] = vmin;
                    b[i] = vmin;
                    break;
                case 2:
                    a[i] = vmin;
                    b[i] = vmax;
                    break;
                case 3:
                    a[i] = vmax;
                    b[i] = static_cast<value_type>(is_signed ? -1 : vmax);
                    break;
                case 4:
                    a[i] = static_cast<value_type>(is_signed ? -1 : vmax);
                    b[i] = static_cast<value_type>(is_signed ? -1 : vmax);
                    break;
                case 5:
                    a[i] = vmin;
                    b[i] = static_cast<value_type>(is_signed ? -1 : 1);
                    break;
                case 6:
                    a[i] = static_cast<value_type>(vmax / 2 + 1);
                    b[i] = static_cast<value_type>(vmax / 2 + 1);
                    break;
                case 7:
                    a[i] = static_cast<value_type>(1);
                    b[i] = vmax;
                    break;
                }
            }
            run_case(a, b, "extremes");
        }

        // Pattern B: high-half-non-zero, mixed signs (each lane unique so we
        // catch lane-wise bugs in 32/64-bit emulated mulhi paths).
        {
            array_type a, b;
            constexpr std::size_t bits = 8 * sizeof(value_type);
            constexpr std::size_t half = bits / 2;
            const UT half_mask = (static_cast<UT>(1) << half) - 1;
            for (std::size_t i = 0; i < size; ++i)
            {
                // Spread bits across both halves so the product overflows the
                // low half. Use deterministic but lane-varying patterns.
                UT ua = static_cast<UT>((static_cast<UT>(0xA53C97E1ULL) ^ (static_cast<UT>(i) * 0x9E37ULL))
                                        | (static_cast<UT>(i + 1) << half));
                UT ub = static_cast<UT>((static_cast<UT>(0x6BD1F4A7ULL) ^ (static_cast<UT>(i) * 0xC2B5ULL))
                                        | (static_cast<UT>((i * 3) + 1) << half));
                // Make sure both halves are non-zero so the product spans bits.
                if ((ua & half_mask) == 0)
                    ua |= static_cast<UT>(1);
                if ((ub & half_mask) == 0)
                    ub |= static_cast<UT>(1);
                a[i] = static_cast<value_type>(ua);
                b[i] = static_cast<value_type>(ub);
            }
            run_case(a, b, "wide-bits");
        }

        // Pattern C: signed correction terms — only meaningful for signed
        // types but harmless for unsigned (we still check correctness).
        {
            array_type a, b;
            for (std::size_t i = 0; i < size; ++i)
            {
                // Alternate negative * positive and negative * negative for
                // signed; for unsigned this just samples large magnitudes.
                value_type x = static_cast<value_type>(vmax - static_cast<value_type>(i));
                value_type y = static_cast<value_type>(is_signed
                                                           ? (i % 2 == 0 ? -static_cast<value_type>(i + 1)
                                                                         : static_cast<value_type>(i + 1))
                                                           : static_cast<value_type>(vmax - (i * 7)));
                a[i] = x;
                b[i] = y;
            }
            run_case(a, b, "signed-correction");
        }
    }

    void test_mulhilo_impl(std::false_type /*not integral*/) const { }

    void test_mulhilo() const
    {
        test_mulhilo_impl(typename std::is_integral<value_type>::type {});
    }

    void test_saturated_arithmetic() const
    {
        // batch + batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), xsimd::sadd<value_type>);
            batch_type res = xsimd::sadd(batch_lhs(), batch_rhs());
            INFO("sadd(batch, batch)");
            CHECK_BATCH_EQ(res, expected);
        }
#if 0
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
#endif
        // batch - batch
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(), [](value_type x, value_type y)
                           { return xsimd::ssub(x, y); });
            batch_type res = xsimd::ssub(batch_lhs(), batch_rhs());
            INFO("ssub(batch, batch)");
            CHECK_BATCH_EQ(res, expected);
        }
#if 0
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

            std::fill(expected.begin(), expected.end(), false);
            res = batch_lhs() < batch_lhs();
            INFO("batch < (self)");
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

            auto res_neg = batch_lhs() >= scalar;
            INFO("batch >= scalar");
            CHECK_BATCH_EQ(!res_neg, expected);
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

            std::fill(expected.begin(), expected.end(), true);
            res = batch_lhs() <= batch_lhs();
            INFO("batch < (self)");
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

            auto res_neg = batch_lhs() > scalar;
            INFO("batch > scalar");
            CHECK_BATCH_EQ(!res_neg, expected);
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

            std::fill(expected.begin(), expected.end(), false);
            res = batch_lhs() > batch_lhs();
            INFO("batch > (self)");
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

            auto res_neg = batch_lhs() <= scalar;
            INFO("batch <= scalar");
            CHECK_BATCH_EQ(!res_neg, expected);
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

            std::fill(expected.begin(), expected.end(), true);
            res = batch_lhs() >= batch_lhs();
            INFO("batch >= (self)");
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

            auto res_neg = batch_lhs() < scalar;
            INFO("batch < scalar");
            CHECK_BATCH_EQ(!res_neg, expected);
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
        // fmas
        {
            array_type expected;
            for (std::size_t i = 0; i < expected.size(); ++i)
            {
                // even lanes: x*y - z, odd lanes: x*y + z
                expected[i] = (i & 1u) == 0
                    ? lhs[i] * rhs[i] - rhs[i]
                    : lhs[i] * rhs[i] + rhs[i];
            }
            batch_type res = fmas(batch_lhs(), batch_rhs(), batch_rhs());
            INFO("fmas");
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

    void test_avg() const
    {
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r) -> value_type
                           {
                               if (std::is_integral<value_type>::value)
                               {
                                   return static_cast<value_type>(((long long)l + r) / 2);
                               }
                               else
                               {
                                   return (l + r) / 2;
                               }
                           });
            batch_type res = avg(batch_lhs(), batch_rhs());
            INFO("avg");
            CHECK_BATCH_EQ(res, expected);
        }
        {
            array_type expected;
            std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), expected.begin(),
                           [](const value_type& l, const value_type& r) -> value_type
                           {
                               if (std::is_integral<value_type>::value)
                               {
                                   return static_cast<value_type>(((long long)l + r) / 2 + ((long long)(l + r) & 1));
                               }
                               else
                               {
                                   return (l + r) / 2;
                               }
                           });
            batch_type res = avgr(batch_lhs(), batch_rhs());
            INFO("avgr");
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
        // reduce_mul
        {
            value_type expected = std::accumulate(lhs.cbegin(), lhs.cend(), value_type(1), std::multiplies<value_type>());
            value_type res = reduce_mul(batch_lhs());
            INFO("reduce_mul");
            CHECK_SCALAR_EQ(res, expected);
        }
    }

    template <size_t N>
    std::enable_if_t<4 <= N> test_common_horizontal_operations(std::integral_constant<size_t, N>) const
    {
        // reduce common
        {
            value_type expected = std::accumulate(lhs.cbegin(), lhs.cend(), value_type(1), std::multiplies<value_type>());
            value_type res = reduce(xsimd::mul<typename B::value_type, typename B::arch_type>, batch_lhs());
            INFO("common reduce");
            CHECK_SCALAR_EQ(res, expected);
        }
    }
    void test_common_horizontal_operations(...) const { }

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

    void init_operands()
    {
        XSIMD_IF_CONSTEXPR(std::is_integral<value_type>::value)
        {
            for (size_t i = 0; i < size; ++i)
            {
                bool negative_lhs = std::is_signed<value_type>::value && (i % 2 == 1);
                lhs[i] = value_type(i) * (negative_lhs ? -3 : 3);
                if (lhs[i] == value_type(0))
                {
                    lhs[i] += value_type(1);
                }
                rhs[i] = value_type(i) + value_type(2);
            }
            scalar = value_type(3);
        }
        else
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

    SUBCASE("first element")
    {
        Test.test_first_element();
    }

    SUBCASE("get")
    {
        Test.test_get();
    }

    SUBCASE("arithmetic")
    {
        Test.test_arithmetic();
    }

    SUBCASE("incr decr")
    {
        Test.test_incr_decr();
    }

    SUBCASE("mulhilo")
    {
        Test.test_mulhilo();
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

    SUBCASE("avg")
    {
        Test.test_avg();
    }

    SUBCASE("horizontal_operations")
    {
        Test.test_horizontal_operations();
        Test.test_common_horizontal_operations(std::integral_constant<size_t, sizeof(typename B::value_type)>());
    }

    SUBCASE("boolean_conversions")
    {
        Test.test_boolean_conversions();
    }
}
#endif
