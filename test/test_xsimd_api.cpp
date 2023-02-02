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

#include "xsimd/types/xsimd_utils.hpp"
#include "xsimd/xsimd.hpp"

#include "doctest/doctest.h"

template <class T>
struct scalar_type
{
    using type = T;
};
template <class T, class A>
struct scalar_type<xsimd::batch<T, A>>
{
    using type = T;
};

template <class T>
T extract(T const& value) { return value; }

template <class T, class A>
T extract(xsimd::batch<T, A> const& batch) { return batch.get(0); }

template <class T, class A>
bool extract(xsimd::batch_bool<T, A> const& batch) { return batch.get(0); }

/*
 * Type series
 */

#define INTEGRAL_TYPES_HEAD char, unsigned char, signed char, short, unsigned short, int, unsigned int, long, unsigned long
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
#define INTEGRAL_TYPES_TAIL
#else
#define INTEGRAL_TYPES_TAIL , xsimd::batch<char>, xsimd::batch<unsigned char>, xsimd::batch<signed char>, xsimd::batch<short>, xsimd::batch<unsigned short>, xsimd::batch<int>, xsimd::batch<unsigned int>, xsimd::batch<long>, xsimd::batch<unsigned long>
#endif

#define INTEGRAL_TYPES INTEGRAL_TYPES_HEAD INTEGRAL_TYPES_TAIL

//

#define FLOAT_TYPES_HEAD float, double

#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
#define FLOAT_TYPES_MIDDLE
#define FLOAT_TYPES_TAIL
#else
#define FLOAT_TYPES_MIDDLE , xsimd::batch<float>
#if !XSIMD_WITH_NEON || XSIMD_WITH_NEON64
#define FLOAT_TYPES_TAIL , xsimd::batch<double>
#else
#define FLOAT_TYPES_TAIL
#endif
#endif
#define FLOAT_TYPES FLOAT_TYPES_HEAD FLOAT_TYPES_MIDDLE FLOAT_TYPES_TAIL

//
#define SCALAR_TYPES INTEGRAL_TYPES, FLOAT_TYPES

//
#define ALL_FLOATING_POINT_TYPES_HEAD float, double, std::complex<float>, std::complex<double>

#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
#define ALL_FLOATING_POINT_TYPES_MIDDLE
#define ALL_FLOATING_POINT_TYPES_TAIL
#else
#define ALL_FLOATING_POINT_TYPES_MIDDLE , xsimd::batch<float>, xsimd::batch<std::complex<float>>

#if !XSIMD_WITH_NEON || XSIMD_WITH_NEON64
#define ALL_FLOATING_POINT_TYPES_TAIL , xsimd::batch<double>, xsimd::batch<std::complex<double>>
#else
#define ALL_FLOATING_POINT_TYPES_TAIL
#endif
#endif
#define ALL_FLOATING_POINT_TYPES ALL_FLOATING_POINT_TYPES_HEAD ALL_FLOATING_POINT_TYPES_MIDDLE ALL_FLOATING_POINT_TYPES_TAIL

//

#define COMPLEX_TYPES ALL_FLOATING_POINT_TYPES

//
#define ALL_INTEGRAL_SIGNED_TYPES_HEAD signed char, short, int, long
#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
#define ALL_INTEGRAL_SIGNED_TYPES_TAIL
#else
#define ALL_INTEGRAL_SIGNED_TYPES_TAIL , xsimd::batch<signed char>, xsimd::batch<short>, xsimd::batch<int>, xsimd::batch<long>
#endif

#define ALL_SIGNED_TYPES ALL_INTEGRAL_SIGNED_TYPES_HEAD ALL_INTEGRAL_SIGNED_TYPES_TAIL, ALL_FLOATING_POINT_TYPES

//

#define ALL_TYPES INTEGRAL_TYPES, ALL_FLOATING_POINT_TYPES

/*
 * Functions that apply on scalar types only
 */

template <typename T>
struct xsimd_api_scalar_types_functions
{
    using value_type = typename scalar_type<T>::type;

    void test_bitofsign()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::bitofsign(T(val))), val < 0);
    }

    void test_bitwise_and()
    {
        value_type val0(1);
        value_type val1(3);
        xsimd::as_unsigned_integer_t<value_type> ival0, ival1, ir;
        std::memcpy((void*)&ival0, (void*)&val0, sizeof(val0));
        std::memcpy((void*)&ival1, (void*)&val1, sizeof(val1));
        value_type r;
        ir = ival0 & ival1;
        std::memcpy((void*)&r, (void*)&ir, sizeof(ir));
        CHECK_EQ(extract(xsimd::bitwise_and(T(val0), T(val1))), r);
    }

    void test_bitwise_andnot()
    {
        value_type val0(1);
        value_type val1(3);
        xsimd::as_unsigned_integer_t<value_type> ival0, ival1, ir;
        std::memcpy((void*)&ival0, (void*)&val0, sizeof(val0));
        std::memcpy((void*)&ival1, (void*)&val1, sizeof(val1));
        value_type r;
        ir = ival0 & ~ival1;
        std::memcpy((void*)&r, (void*)&ir, sizeof(ir));
        CHECK_EQ(extract(xsimd::bitwise_andnot(T(val0), T(val1))), r);
    }

    void test_bitwise_not()
    {
        value_type val(1);
        xsimd::as_unsigned_integer_t<value_type> ival, ir;
        std::memcpy((void*)&ival, (void*)&val, sizeof(val));
        value_type r;
        ir = ~ival;
        std::memcpy((void*)&r, (void*)&ir, sizeof(ir));
        CHECK_EQ(extract(xsimd::bitwise_not(T(val))), r);
    }

    void test_bitwise_or()
    {
        value_type val0(1);
        value_type val1(4);
        xsimd::as_unsigned_integer_t<value_type> ival0, ival1, ir;
        std::memcpy((void*)&ival0, (void*)&val0, sizeof(val0));
        std::memcpy((void*)&ival1, (void*)&val1, sizeof(val1));
        value_type r;
        ir = ival0 | ival1;
        std::memcpy((void*)&r, (void*)&ir, sizeof(ir));
        CHECK_EQ(extract(xsimd::bitwise_or(T(val0), T(val1))), r);
    }

    void test_bitwise_xor()
    {
        value_type val0(1);
        value_type val1(2);
        xsimd::as_unsigned_integer_t<value_type> ival0, ival1, ir;
        std::memcpy((void*)&ival0, (void*)&val0, sizeof(val0));
        std::memcpy((void*)&ival1, (void*)&val1, sizeof(val1));
        value_type r;
        ir = ival0 ^ ival1;
        std::memcpy((void*)&r, (void*)&ir, sizeof(ir));
        CHECK_EQ(extract(xsimd::bitwise_xor(T(val0), T(val1))), r);
    }

    void test_clip()
    {
        value_type val0(5);
        value_type val1(2);
        value_type val2(3);
        CHECK_EQ(extract(xsimd::clip(T(val0), T(val1), T(val2))), val0 <= val1 ? val1 : (val0 >= val2 ? val2 : val0));
    }

    void test_ge()
    {
        value_type val0(1);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::ge(T(val0), T(val1))), val0 >= val1);
    }

    void test_gt()
    {
        value_type val0(1);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::gt(T(val0), T(val1))), val0 > val1);
    }

    void test_le()
    {
        value_type val0(1);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::le(T(val0), T(val1))), val0 <= val1);
    }

    void test_lt()
    {
        value_type val0(1);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::lt(T(val0), T(val1))), val0 < val1);
    }

    void test_max()
    {
        value_type val0(1);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::max(T(val0), T(val1))), std::max(val0, val1));
    }

    void test_min()
    {
        value_type val0(1);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::min(T(val0), T(val1))), std::min(val0, val1));
    }

    void test_remainder()
    {
        value_type val0(1);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::remainder(T(val0), T(val1))), val0 - xsimd::as_integer_t<value_type>(val0) / xsimd::as_integer_t<value_type>(val1));
    }
    void test_sign()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::sign(T(val))), val == 0 ? 0 : val > 0 ? 1
                                                                      : -1);
    }
    void test_signnz()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::signnz(T(val))), val == 0 ? 1 : val > 0 ? 1
                                                                        : -1);
    }
};

TEST_CASE_TEMPLATE("[xsimd api | scalar types]", B, SCALAR_TYPES)
{
    xsimd_api_scalar_types_functions<B> Test;
    SUBCASE("bitofsign")
    {
        Test.test_bitofsign();
    }

    SUBCASE("bitwise_and")
    {
        Test.test_bitwise_and();
    }

    SUBCASE("bitwise_andnot")
    {
        Test.test_bitwise_andnot();
    }

    SUBCASE("bitwise_not")
    {
        Test.test_bitwise_not();
    }

    SUBCASE("bitwise_or")
    {
        Test.test_bitwise_or();
    }

    SUBCASE("bitwise_xor")
    {
        Test.test_bitwise_xor();
    }

    SUBCASE("clip")
    {
        Test.test_clip();
    }

    SUBCASE("ge")
    {
        Test.test_ge();
    }

    SUBCASE("gt")
    {
        Test.test_gt();
    }

    SUBCASE("le")
    {
        Test.test_le();
    }

    SUBCASE("lt")
    {
        Test.test_lt();
    }

    SUBCASE("max")
    {
        Test.test_max();
    }

    SUBCASE("min")
    {
        Test.test_min();
    }

    SUBCASE("remainder")
    {
        Test.test_remainder();
    }

    SUBCASE("sign")
    {
        Test.test_sign();
    }

    SUBCASE("signnz")
    {
        Test.test_signnz();
    }
}

/*
 * Functions that apply on integral types only
 */

template <typename T>
struct xsimd_api_integral_types_functions
{
    using value_type = typename scalar_type<T>::type;

    void test_mod()
    {
        value_type val0(5);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::mod(T(val0), T(val1))), val0 % val1);
    }
    void test_sadd()
    {
        value_type val0(122);
        value_type val1(std::numeric_limits<value_type>::max());
        CHECK_EQ(extract(xsimd::sadd(T(val0), T(val1))), (val0 > std::numeric_limits<value_type>::max() - val1) ? std::numeric_limits<value_type>::max() : (val0 + val1));
    }
    void test_ssub()
    {
        value_type val0(122);
        value_type val1(121);
        CHECK_EQ(extract(xsimd::ssub(T(val0), T(val1))), (val0 < std::numeric_limits<value_type>::min() + val1) ? std::numeric_limits<value_type>::min() : (val0 - val1));
    }
};

TEST_CASE_TEMPLATE("[xsimd api | integral types functions]", B, INTEGRAL_TYPES)
{
    xsimd_api_integral_types_functions<B> Test;

    SUBCASE("mod")
    {
        Test.test_mod();
    }

    SUBCASE("sadd")
    {
        Test.test_sadd();
    }

    SUBCASE("ssub")
    {
        Test.test_ssub();
    }
}

/*
 * Functions that apply on floating points types only
 */

template <typename T>
struct xsimd_api_float_types_functions
{
    using value_type = typename scalar_type<T>::type;

    void test_acos()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::acos(T(val))), std::acos(val));
    }
    void test_acosh()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::acosh(T(val))), std::acosh(val));
    }
    void test_asin()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::asin(T(val))), std::asin(val));
    }
    void test_asinh()
    {
        value_type val(0);
        CHECK_EQ(extract(xsimd::asinh(T(val))), std::asinh(val));
    }
    void test_atan()
    {
        value_type val(0);
        CHECK_EQ(extract(xsimd::atan(T(val))), std::atan(val));
    }
    void test_atan2()
    {
        value_type val0(0);
        value_type val1(1);
        CHECK_EQ(extract(xsimd::atan2(T(val0), T(val1))), std::atan2(val0, val1));
    }
    void test_atanh()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::atanh(T(val))), std::atanh(val));
    }
    void test_cbrt()
    {
        value_type val(8);
        CHECK_EQ(extract(xsimd::cbrt(T(val))), std::cbrt(val));
    }
    void test_ceil()
    {
        value_type val(1.5);
        CHECK_EQ(extract(xsimd::ceil(T(val))), std::ceil(val));
    }

    void test_copysign()
    {
        value_type val0(2);
        value_type val1(-1);
        CHECK_EQ(extract(xsimd::copysign(T(val0), T(val1))), (value_type)std::copysign(val0, val1));
    }
    void test_cos()
    {
        value_type val(0);
        CHECK_EQ(extract(xsimd::cos(T(val))), std::cos(val));
    }
    void test_cosh()
    {
        value_type val(0);
        CHECK_EQ(extract(xsimd::cosh(T(val))), std::cosh(val));
    }
    void test_exp()
    {
        value_type val(2);
        CHECK_EQ(extract(xsimd::exp(T(val))), std::exp(val));
    }
    void test_exp10()
    {
        value_type val(2);
        CHECK_EQ(extract(xsimd::exp10(T(val))), std::pow(value_type(10), val));
    }
    void test_exp2()
    {
        value_type val(2);
        CHECK_EQ(extract(xsimd::exp2(T(val))), std::exp2(val));
    }
    void test_expm1()
    {
        value_type val(2);
        CHECK_EQ(extract(xsimd::expm1(T(val))), std::expm1(val));
    }
    void test_erf()
    {
        value_type val(2);
        CHECK_EQ(extract(xsimd::erf(T(val))), std::erf(val));
    }
    void test_erfc()
    {
        // FIXME: can we do better?
        value_type val(0);
        CHECK_EQ(extract(xsimd::erfc(T(val))), doctest::Approx(std::erfc(val)).epsilon(10e-7));
    }
    void test_fabs()
    {
        value_type val(-3);
        CHECK_EQ(extract(xsimd::fabs(T(val))), std::abs(val));
    }
    void test_fdim()
    {
        value_type val0(-3);
        value_type val1(1);
        CHECK_EQ(extract(xsimd::fdim(T(val0), T(val1))), std::fdim(val0, val1));
    }
    void test_floor()
    {
        value_type val(3.1);
        CHECK_EQ(extract(xsimd::floor(T(val))), std::floor(val));
    }
    void test_fmax()
    {
        value_type val0(3);
        value_type val1(1);
        CHECK_EQ(extract(xsimd::fmax(T(val0), T(val1))), std::fmax(val0, val1));
    }
    void test_fmin()
    {
        value_type val0(3);
        value_type val1(1);
        CHECK_EQ(extract(xsimd::fmin(T(val0), T(val1))), std::fmin(val0, val1));
    }
    void test_fmod()
    {
        value_type val0(3);
        value_type val1(1);
        CHECK_EQ(extract(xsimd::fmin(T(val0), T(val1))), std::fmin(val0, val1));
    }
    void test_frexp()
    {
        value_type val(3.3);
        int res;
        typename std::conditional<std::is_floating_point<T>::value, int, xsimd::as_integer_t<T>>::type vres;
        CHECK_EQ(extract(xsimd::frexp(T(val), vres)), std::frexp(val, &res));
        CHECK_EQ(extract(vres), res);
    }
    void test_hypot()
    {
        value_type val0(3);
        value_type val1(1);
        CHECK_EQ(extract(xsimd::hypot(T(val0), T(val1))), std::hypot(val0, val1));
    }
    void test_is_even()
    {
        value_type val(4);
        CHECK_EQ(extract(xsimd::is_even(T(val))), (val == long(val)) && (long(val) % 2 == 0));
    }
    void test_is_flint()
    {
        value_type val(4.1);
        CHECK_EQ(extract(xsimd::is_flint(T(val))), (val == long(val)));
    }
    void test_is_odd()
    {
        value_type val(4);
        CHECK_EQ(extract(xsimd::is_odd(T(val))), (val == long(val)) && (long(val) % 2 == 1));
    }
    void test_isinf()
    {
        value_type val(4);
        CHECK_EQ(extract(xsimd::isinf(T(val))), std::isinf(val));
    }
    void test_isfinite()
    {
        value_type val(4);
        CHECK_EQ(extract(xsimd::isfinite(T(val))), std::isfinite(val));
    }
    void test_isnan()
    {
        value_type val(4);
        CHECK_EQ(extract(xsimd::isnan(T(val))), std::isnan(val));
    }
    void test_ldexp()
    {
        value_type val0(4);
        xsimd::as_integer_t<value_type> val1(2);
        CHECK_EQ(extract(xsimd::ldexp(T(val0), xsimd::as_integer_t<T>(val1))), std::ldexp(val0, val1));
    }
    void test_lgamma()
    {
        value_type val(2);
        CHECK_EQ(extract(xsimd::lgamma(T(val))), std::lgamma(val));
    }
    void test_log()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::log(T(val))), std::log(val));
    }

    void test_log2()
    {
        value_type val(2);
        CHECK_EQ(extract(xsimd::log2(T(val))), std::log2(val));
    }

    void test_log10()
    {
        value_type val(10);
        CHECK_EQ(extract(xsimd::log10(T(val))), std::log10(val));
    }

    void test_log1p()
    {
        value_type val(0);
        CHECK_EQ(extract(xsimd::log1p(T(val))), std::log1p(val));
    }
    void test_nearbyint()
    {
        value_type val(3.1);
        CHECK_EQ(extract(xsimd::nearbyint(T(val))), std::nearbyint(val));
    }
    void test_nearbyint_as_int()
    {
        value_type val(3.1);
        CHECK_EQ(extract(xsimd::nearbyint_as_int(T(val))), long(std::nearbyint(val)));
    }
    void test_nextafter()
    {
        value_type val0(3);
        value_type val1(4);
        CHECK_EQ(extract(xsimd::nextafter(T(val0), T(val1))), std::nextafter(val0, val1));
    }
    void test_polar()
    {
        value_type val0(3);
        value_type val1(4);
        CHECK_EQ(extract(xsimd::polar(T(val0), T(val1))), std::polar(val0, val1));
    }
    void test_pow()
    {
        value_type val0(2);
        value_type val1(2);
        int ival1 = 4;
        CHECK_EQ(extract(xsimd::pow(T(val0), T(val1))), std::pow(val0, val1));
        CHECK_EQ(extract(xsimd::pow(T(val0), ival1)), std::pow(val0, ival1));
    }
    void test_reciprocal()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::reciprocal(T(val))), doctest::Approx(value_type(1) / val).epsilon(10e-2));
    }
    void test_rint()
    {
        value_type val(3.1);
        CHECK_EQ(extract(xsimd::rint(T(val))), std::rint(val));
    }
    void test_round()
    {
        value_type val(3.1);
        CHECK_EQ(extract(xsimd::round(T(val))), std::round(val));
    }
    void test_rsqrt()
    {
        value_type val(4);
        CHECK_EQ(extract(xsimd::rsqrt(T(val))), doctest::Approx(value_type(1) / std::sqrt(val)).epsilon(10e-4));
    }
    void test_sin()
    {
        value_type val(0);
        CHECK_EQ(extract(xsimd::sin(T(val))), std::sin(val));
    }
    void test_sincos()
    {
        value_type val(0);
        auto vres = xsimd::sincos(T(val));
        CHECK_EQ(extract(vres.first), std::sin(val));
        CHECK_EQ(extract(vres.second), std::cos(val));
    }
    void test_sinh()
    {
        value_type val(0);
        CHECK_EQ(extract(xsimd::sinh(T(val))), std::sinh(val));
    }
    void test_sqrt()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::sqrt(T(val))), std::sqrt(val));
    }
    void test_tan()
    {
        value_type val(0);
        CHECK_EQ(extract(xsimd::tan(T(val))), std::tan(val));
    }
    void test_tanh()
    {
        value_type val(0);
        CHECK_EQ(extract(xsimd::tanh(T(val))), std::tanh(val));
    }
    void test_tgamma()
    {
        value_type val(2);
        CHECK_EQ(extract(xsimd::tgamma(T(val))), std::tgamma(val));
    }
    void test_trunc()
    {
        value_type val(2.1);
        CHECK_EQ(extract(xsimd::trunc(T(val))), std::trunc(val));
    }
};

TEST_CASE_TEMPLATE("[xsimd api | float types functions]", B, FLOAT_TYPES)
{
    xsimd_api_float_types_functions<B> Test;

    SUBCASE("acos")
    {
        Test.test_acos();
    }

    SUBCASE("acosh")
    {
        Test.test_acosh();
    }

    SUBCASE("asin")
    {
        Test.test_asin();
    }

    SUBCASE("asinh")
    {
        Test.test_asinh();
    }

    SUBCASE("atan")
    {
        Test.test_atan();
    }

    SUBCASE("atan2")
    {
        Test.test_atan2();
    }

    SUBCASE("atanh")
    {
        Test.test_atanh();
    }

    SUBCASE("cbrt")
    {
        Test.test_cbrt();
    }

    SUBCASE("ceil")
    {
        Test.test_ceil();
    }

    SUBCASE("copysign")
    {
        Test.test_copysign();
    }

    SUBCASE("cos")
    {
        Test.test_cos();
    }

    SUBCASE("cosh")
    {
        Test.test_cosh();
    }

    SUBCASE("exp")
    {
        Test.test_exp();
    }

    SUBCASE("exp10")
    {
        Test.test_exp10();
    }

    SUBCASE("exp2")
    {
        Test.test_exp2();
    }

    SUBCASE("expm1")
    {
        Test.test_expm1();
    }

    SUBCASE("erf")
    {
        Test.test_erf();
    }

    SUBCASE("erfc")
    {
        Test.test_erfc();
    }

    SUBCASE("fabs")
    {
        Test.test_fabs();
    }

    SUBCASE("fdim")
    {
        Test.test_fdim();
    }

    SUBCASE("floor")
    {
        Test.test_floor();
    }

    SUBCASE("fmax")
    {
        Test.test_fmax();
    }

    SUBCASE("fmin")
    {
        Test.test_fmin();
    }

    SUBCASE("fmod")
    {
        Test.test_fmod();
    }
    SUBCASE("frexp")
    {
        Test.test_frexp();
    }
    SUBCASE("hypot")
    {
        Test.test_hypot();
    }
    SUBCASE("is_even")
    {
        Test.test_is_even();
    }
    SUBCASE("is_flint")
    {
        Test.test_is_flint();
    }
    SUBCASE("is_odd")
    {
        Test.test_is_odd();
    }
    SUBCASE("isinf")
    {
        Test.test_isinf();
    }
    SUBCASE("isfinite")
    {
        Test.test_isfinite();
    }
    SUBCASE("isnan")
    {
        Test.test_isnan();
    }
    SUBCASE("ldexp")
    {
        Test.test_ldexp();
    }
    SUBCASE("lgamma")
    {
        Test.test_lgamma();
    }

    SUBCASE("log")
    {
        Test.test_log();
    }

    SUBCASE("log2")
    {
        Test.test_log2();
    }

    SUBCASE("log10")
    {
        Test.test_log10();
    }

    SUBCASE("log1p")
    {
        Test.test_log1p();
    }

    SUBCASE("nearbyint")
    {
        Test.test_nearbyint();
    }

    SUBCASE("nearbyint_as_int")
    {
        Test.test_nearbyint_as_int();
    }

    SUBCASE("nextafter")
    {
        Test.test_nextafter();
    }

    SUBCASE("polar")
    {
        Test.test_polar();
    }

    SUBCASE("pow")
    {
        Test.test_pow();
    }

    SUBCASE("reciprocal")
    {
        Test.test_reciprocal();
    }

    SUBCASE("rint")
    {
        Test.test_rint();
    }

    SUBCASE("round")
    {
        Test.test_round();
    }

    SUBCASE("rsqrt")
    {
        Test.test_rsqrt();
    }

    SUBCASE("sin")
    {
        Test.test_sin();
    }

    SUBCASE("sincos")
    {
        Test.test_sincos();
    }

    SUBCASE("sinh")
    {
        Test.test_sinh();
    }

    SUBCASE("sqrt")
    {
        Test.test_sqrt();
    }

    SUBCASE("tan")
    {
        Test.test_tan();
    }

    SUBCASE("tanh")
    {
        Test.test_tanh();
    }

    SUBCASE("tgamma")
    {
        Test.test_tgamma();
    }

    SUBCASE("trunc")
    {
        Test.test_trunc();
    }
}

/*
 * Functions that apply on complex and floating point types only
 */

template <typename T>
struct xsimd_api_complex_types_functions
{
    using value_type = typename scalar_type<T>::type;

    void test_arg()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::arg(T(val))), std::arg(val));
    }

    void test_conj()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::conj(T(val))), std::conj(val));
    }

    void test_norm()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::norm(T(val))), std::norm(val));
    }

    void test_proj()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::proj(T(val))), std::proj(val));
    }
};

TEST_CASE_TEMPLATE("[xsimd api | complex types functions]", B, COMPLEX_TYPES)
{
    xsimd_api_complex_types_functions<B> Test;
    SUBCASE("arg")
    {
        Test.test_arg();
    }

    SUBCASE("conj")
    {
        Test.test_conj();
    }

    SUBCASE("norm")
    {
        Test.test_norm();
    }

    SUBCASE("proj")
    {
        Test.test_proj();
    }
}

/*
 * Functions that apply on all signed types
 */
template <typename T>
struct xsimd_api_all_signed_types_functions
{
    using value_type = typename scalar_type<T>::type;

    void test_abs()
    {
        value_type val(-1);
        CHECK_EQ(extract(xsimd::abs(T(val))), std::abs(val));
    }

    void test_fnms()
    {
        value_type val0(1);
        value_type val1(3);
        value_type val2(5);
        CHECK_EQ(extract(xsimd::fnms(T(val0), T(val1), T(val2))), -(val0 * val1) - val2);
    }

    void test_neg()
    {
        value_type val(-1);
        CHECK_EQ(extract(xsimd::neg(T(val))), -val);
    }
};

TEST_CASE_TEMPLATE("[xsimd api | all signed types functions]", B, ALL_SIGNED_TYPES)
{
    xsimd_api_all_signed_types_functions<B> Test;

    SUBCASE("abs")
    {
        Test.test_abs();
    }
    SUBCASE("fnms")
    {
        Test.test_fnms();
    }
    SUBCASE("neg")
    {
        Test.test_neg();
    }
}

/*
 * Functions that apply on all types
 */

template <typename T>
struct xsimd_api_all_types_functions
{
    using value_type = typename scalar_type<T>::type;

    void test_add()
    {
        value_type val0(1);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::add(T(val0), T(val1))), val0 + val1);
    }

    void test_div()
    {
        value_type val0(1);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::div(T(val0), T(val1))), val0 / val1);
    }

    void test_eq()
    {
        value_type val0(1);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::eq(T(val0), T(val1))), val0 == val1);
    }

    void test_fma()
    {
        value_type val0(1);
        value_type val1(3);
        value_type val2(5);
        CHECK_EQ(extract(xsimd::fma(T(val0), T(val1), T(val2))), val0 * val1 + val2);
    }

    void test_fms()
    {
        value_type val0(1);
        value_type val1(5);
        value_type val2(3);
        CHECK_EQ(extract(xsimd::fms(T(val0), T(val1), T(val2))), val0 * val1 - val2);
    }

    void test_fnma()
    {
        value_type val0(1);
        value_type val1(3);
        value_type val2(5);
        CHECK_EQ(extract(xsimd::fnma(T(val0), T(val1), T(val2))), -(val0 * val1) + val2);
    }

    void test_mul()
    {
        value_type val0(2);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::mul(T(val0), T(val1))), val0 * val1);
    }
    void test_neq()
    {
        value_type val0(1);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::neq(T(val0), T(val1))), val0 != val1);
    }
    void test_pos()
    {
        value_type val(1);
        CHECK_EQ(extract(xsimd::pos(T(val))), +val);
    }
    void test_select()
    {
        value_type val0(2);
        value_type val1(3);
        CHECK_EQ(extract(xsimd::select(T(val0) != T(val1), T(val0), T(val1))), val0 != val1 ? val0 : val1);
    }
    void test_sub()
    {
        value_type val0(3);
        value_type val1(2);
        CHECK_EQ(extract(xsimd::sub(T(val0), T(val1))), val0 - val1);
    }
};

TEST_CASE_TEMPLATE("[xsimd api | all types functions]", B, ALL_TYPES)
{
    xsimd_api_all_types_functions<B> Test;

    SUBCASE("add")
    {
        Test.test_add();
    }

    SUBCASE("div")
    {
        Test.test_div();
    }

    SUBCASE("eq")
    {
        Test.test_eq();
    }

    SUBCASE("fma")
    {
        Test.test_fma();
    }

    SUBCASE("fms")
    {
        Test.test_fms();
    }

    SUBCASE("fnma")
    {
        Test.test_fnma();
    }

    SUBCASE("mul")
    {
        Test.test_mul();
    }

    SUBCASE("neq")
    {
        Test.test_neq();
    }

    SUBCASE("pos")
    {
        Test.test_pos();
    }
    SUBCASE("select")
    {
        Test.test_select();
    }
    SUBCASE("sub")
    {
        Test.test_sub();
    }
}

/*
 * Functions that apply only to floating point types
 */
template <typename T>
struct xsimd_api_all_floating_point_types_functions
{
    using value_type = typename scalar_type<T>::type;

    void test_neq_nan()
    {
        value_type valNaN(std::numeric_limits<value_type>::signaling_NaN());
        value_type val1(1.0);
        CHECK_EQ(extract(xsimd::neq(T(valNaN), T(val1))), valNaN != val1);
    }
};

TEST_CASE_TEMPLATE("[xsimd api | all floating point types functions]", B, ALL_FLOATING_POINT_TYPES)
{
    xsimd_api_all_floating_point_types_functions<B> Test;
    Test.test_neq_nan();
}
