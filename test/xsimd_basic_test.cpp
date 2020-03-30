/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <complex>
#include <fstream>
#include <iostream>
#include <map>

#include "gtest/gtest.h"

#include "xsimd_basic_test.hpp"
#include "xsimd_complex_basic_test.hpp"

namespace xsimd
{
    template <class T, size_t N, size_t A>
    bool test_simd(std::ostream& out, const std::string& name)
    {
        simd_basic_tester<T, N, A> tester(name);
        return test_simd_basic(out, tester);
    }

    template <class T, size_t N, size_t A>
    bool test_simd_complex(std::ostream& out, const std::string& name)
    {
        simd_complex_basic_tester<T, N, A> tester(name);
        return test_simd_complex_basic(out, tester);
    }

    template <class T, size_t N, size_t A>
    bool test_simd_int(std::ostream& out, const std::string& name)
    {
        simd_int_basic_tester<T, N, A> tester(name);
        return test_simd_int_basic(out, tester);
    }

    template <size_t N, size_t A>
    bool test_simd_convert(std::ostream& out, const std::string& name)
    {
        simd_convert_tester<N, A> tester(name);
        return test_simd_conversion(out, tester);
    }

    template <size_t N, size_t A>
    bool test_simd_batch_cast(std::ostream& out, const std::string& name)
    {
        simd_batch_cast_tester<N, A> tester(name);
        bool res = true;
        res &= test_simd_batch_cast(out, tester);
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
        res &= test_simd_batch_cast_sizeshift1(out, tester);
#endif
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512_VERSION
        res &= test_simd_batch_cast_sizeshift2(out, tester);
#endif
        return res;
    }

    template <size_t N, size_t A>
    bool test_simd_bitwise_cast(std::ostream& out, const std::string& name)
    {
        simd_bitwise_cast_tester<N, A> tester(name);
        return test_simd_bitwise_cast(out, tester);
    }

    template <class T, size_t N, size_t A>
    bool test_simd_load_store(std::ostream& out, const std::string& name)
    {
        simd_load_store_tester<T, N, A> tester(name);
        return test_simd_load_store(out, tester);
    }

    template <class T, size_t N, size_t A>
    bool test_complex_simd_load_store(std::ostream& out, const std::string& name)
    {
        simd_complex_ls_tester<T, N, A> tester(name);
        return test_complex_simd_load_store(out, tester);
    }

    // No batch exists for this type, that's required
    // for testing simd_return_type
    struct fake_scalar_type
    {
    };

    struct res_checker
    {
        template <class T1, class T2>
        simd_return_type<T1, T2> load_simd() const
        {
            return simd_return_type<T1, T2>();
        }
    };

    template <class... T>
    struct make_res_wrapper
    {
        using type = void;
    };

    template <class... T>
    using res_wrapper = typename make_res_wrapper<T...>::type;

    template <class T1, class T2, class = res_wrapper<>>
    struct check_return_type
    {
    public:
        static constexpr bool value() { return m_value; }
    private:
        static constexpr bool m_value = false;
    };

    template <class T1, class T2>
    struct check_return_type<T1, T2, res_wrapper<decltype(std::declval<res_checker>().template load_simd<T1, T2>())>>
    {
    public:
        static constexpr bool value() { return m_value; }
    private:
        static constexpr bool m_value = true;
    };
}

TEST(xsimd, simd_return_type)
{
    EXPECT_TRUE((xsimd::check_return_type<double, double>::value()));
    EXPECT_TRUE((xsimd::check_return_type<std::complex<double>, double>::value()));
}

/*******************
 * sse basic tests *
 *******************/

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
TEST(xsimd, sse_float_basic)
{
    std::ofstream out("log/sse_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<float, 4, 16>(out, "sse float");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_double_basic)
{
    std::ofstream out("log/sse_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<double, 2, 16>(out, "sse double");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_int8_basic)
{
    std::ofstream out("log/sse_int8_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int8_t, 16, 16>(out, "sse int8");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_int8_load_store)
{
    std::ofstream out("log/sse_int8_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int8_t, 16, 16>(out, "sse int8");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_uint8_basic)
{
    std::ofstream out("log/sse_uint8_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint8_t, 16, 16>(out, "sse uint8");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_uint8_load_store)
{
    std::ofstream out("log/sse_uint8_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint8_t, 16, 16>(out, "sse uint8");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_int16_basic)
{
    std::ofstream out("log/sse_int16_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int16_t, 8, 16>(out, "sse int16");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_int16_load_store)
{
    std::ofstream out("log/sse_int16_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int16_t, 8, 16>(out, "sse int16");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_uint16_basic)
{
    std::ofstream out("log/sse_uint16_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint16_t, 8, 16>(out, "sse uint16");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_uint16_load_store)
{
    std::ofstream out("log/sse_uint16_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint16_t, 8, 16>(out, "sse uint16");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_int32_basic)
{
    std::ofstream out("log/sse_int32_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int32_t, 4, 16>(out, "sse int32");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_int32_load_store)
{
    std::ofstream out("log/sse_int32_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int32_t, 4, 16>(out, "sse int32");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_uint32_basic)
{
    std::ofstream out("log/sse_uint32_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint32_t, 4, 16>(out, "sse uint32");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_uint32_load_store)
{
    std::ofstream out("log/sse_uint32_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint32_t, 4, 16>(out, "sse uint32");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_int64_basic)
{
    std::ofstream out("log/sse_int64_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int64_t, 2, 16>(out, "sse int64");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_int64_load_store)
{
    std::ofstream out("log/sse_int64_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int64_t, 2, 16>(out, "sse int64");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_uint64_basic)
{
    std::ofstream out("log/sse_uint64_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint64_t, 2, 16>(out, "sse uint64");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_uint64_load_store)
{
    std::ofstream out("log/sse_uint64_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint64_t, 2, 16>(out, "sse uint64");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_complex_float_basic)
{
    std::ofstream out("log/sse_complex_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<std::complex<float>, 4, 16>(out, "sse complex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_complex_double_basic)
{
    std::ofstream out("log/sse_complex_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<std::complex<double>, 2, 16>(out, "sse complex double");
    EXPECT_TRUE(res);
}

#ifdef XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, sse_xtl_xcomplex_float_basic)
{
    std::ofstream out("log/sse_xtl_xcomplex_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<xtl::xcomplex<float>, 4, 16>(out, "sse xtl xcomplex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_xtl_xcomplex_double_basic)
{
    std::ofstream out("log/sse_xtl_xcomplex_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<xtl::xcomplex<double>, 2, 16>(out, "sse xtl xcomplex double");
    EXPECT_TRUE(res);
}
#endif

TEST(xsimd, sse_conversion)
{
    std::ofstream out("log/sse_conversion.log", std::ios_base::out);
    bool res = xsimd::test_simd_convert<2, 16>(out, "sse conversion");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_batch_cast)
{
    std::ofstream out("log/sse_batch_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_batch_cast<2, 16>(out, "sse batch cast");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_bitwise_cast)
{
    std::ofstream out("log/sse_bitwise_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_bitwise_cast<2, 16>(out, "sse bitwise cast");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_complex_float_load_store)
{
    std::ofstream out("log/sse_complex_float_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<std::complex<float>, 4, 16>(out, "sse complex float load store");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_complex_double_load_store)
{
    std::ofstream out("log/sse_complex_double_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<std::complex<double>, 2, 16>(out, "sse complex float double store");
    EXPECT_TRUE(res);
}

#ifdef XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, sse_xtl_xcomplex_float_load_store)
{
    std::ofstream out("log/sse_xtl_xcomplex_float_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<xtl::xcomplex<float>, 4, 16>(out, "sse xtl xcomplex float load store");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_xtl_xcomplex_double_load_store)
{
    std::ofstream out("log/sse_xtl_xcomplex_double_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<xtl::xcomplex<double>, 2, 16>(out, "sse xtl xcomplex double load store");
    EXPECT_TRUE(res);
}
#endif
#endif

  /*******************
   * avx basic tests *
   *******************/

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
TEST(xsimd, avx_float_basic)
{
    std::ofstream out("log/avx_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<float, 8, 32>(out, "avx float");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_double_basic)
{
    std::ofstream out("log/avx_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<double, 4, 32>(out, "avx double");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_int8_basic)
{
    std::ofstream out("log/avx_int8_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int8_t, 32, 32>(out, "avx int8");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_int8_load_store)
{
    std::ofstream out("log/avx_int8_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int8_t, 32, 32>(out, "avx int8");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_uint8_basic)
{
    std::ofstream out("log/avx_uint8_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint8_t, 32, 32>(out, "avx uint8");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_uint8_load_store)
{
    std::ofstream out("log/avx_uint8_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint8_t, 32, 32>(out, "avx uint8");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_int16_basic)
{
    std::ofstream out("log/avx_int16_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int16_t, 16, 32>(out, "avx int16");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_int16_load_store)
{
    std::ofstream out("log/avx_int16_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int16_t, 16, 32>(out, "avx int16");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_uint16_basic)
{
    std::ofstream out("log/avx_uint16_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint16_t, 16, 32>(out, "avx uint16");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_uint16_load_store)
{
    std::ofstream out("log/avx_uint16_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint16_t, 16, 32>(out, "avx uint16");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_int32_basic)
{
    std::ofstream out("log/avx_int32_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int32_t, 8, 32>(out, "avx int32");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_int32_load_store)
{
    std::ofstream out("log/avx_int32_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int32_t, 8, 32>(out, "avx int32");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_uint32_basic)
{
    std::ofstream out("log/avx_uint32_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint32_t, 8, 32>(out, "avx uint32");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_uint32_load_store)
{
    std::ofstream out("log/avx_uint32_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint32_t, 8, 32>(out, "avx uint32");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_int64_basic)
{
    std::ofstream out("log/avx_int64_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int64_t, 4, 32>(out, "avx int64");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_int64_load_store)
{
    std::ofstream out("log/avx_int64_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int64_t, 4, 32>(out, "avx int64");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_uint64_basic)
{
    std::ofstream out("log/avx_uint64_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint64_t, 4, 32>(out, "avx uint64");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_uint64_load_store)
{
    std::ofstream out("log/avx_uint64_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint64_t, 4, 32>(out, "avx uint64");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_complex_float_basic)
{
    std::ofstream out("log/avx_complex_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<std::complex<float>, 8, 32>(out, "avx complex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_complex_double_basic)
{
    std::ofstream out("log/avx_complex_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<std::complex<double>, 4, 32>(out, "avx complex double");
    EXPECT_TRUE(res);
}

#ifdef XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, avx_xtl_xcomplex_float_basic)
{
    std::ofstream out("log/avx_xtl_xcomplex_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<xtl::xcomplex<float>, 8, 32>(out, "avx xtl xcomplex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_xtl_xcomplex_double_basic)
{
    std::ofstream out("log/avx_xtl_xcomplex_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<xtl::xcomplex<double>, 4, 32>(out, "avx xtl xcomplex double");
    EXPECT_TRUE(res);
}
#endif

TEST(xsimd, avx_conversion)
{
    std::ofstream out("log/avx_conversion.log", std::ios_base::out);
    bool res = xsimd::test_simd_convert<4, 32>(out, "avx conversion");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_batch_cast)
{
    std::ofstream out("log/avx_batch_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_batch_cast<4, 32>(out, "avx batch cast");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_bitwise_cast)
{
    std::ofstream out("log/avx_bitwise_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_bitwise_cast<4, 32>(out, "avx bitwise cast");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_complex_float_load_store)
{
    std::ofstream out("log/avx_complex_float_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<std::complex<float>, 8, 32>(out, "avx complex float load store");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_complex_double_load_store)
{
    std::ofstream out("log/avx_complex_double_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<std::complex<double>, 4, 32>(out, "avx complex float double store");
    EXPECT_TRUE(res);
}

#ifdef XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, avx_xtl_xcomplex_float_load_store)
{
    std::ofstream out("log/avx_xtl_xcomplex_float_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<xtl::xcomplex<float>, 8, 32>(out, "avx xtl xcomplex float load store");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_xtl_xcomplex_double_load_store)
{
    std::ofstream out("log/avx_xtl_xcomplex_double_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<xtl::xcomplex<double>, 4, 32>(out, "avx xtl xcomplex double load store");
    EXPECT_TRUE(res);
}
#endif
#endif

/**********************
 * avx512 basic tests *
 **********************/

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512_VERSION
TEST(xsimd, avx512_float_basic)
{
    std::ofstream out("log/avx512_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<float, 16, 64>(out, "avx512 float");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_double_basic)
{
    std::ofstream out("log/avx512_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<double, 8, 64>(out, "avx512 double");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_int8_basic)
{
    std::ofstream out("log/avx512_int8_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int8_t, 64, 64>(out, "avx512 int8");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_int8_load_store)
{
    std::ofstream out("log/avx512_int8_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int8_t, 64, 64>(out, "avx512 int8");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_uint8_basic)
{
    std::ofstream out("log/avx512_uint8_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint8_t, 64, 64>(out, "avx512 uint8");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_uint8_load_store)
{
    std::ofstream out("log/avx512_uint8_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint8_t, 64, 64>(out, "avx512 uint8");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_int16_basic)
{
    std::ofstream out("log/avx512_int16_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int16_t, 32, 64>(out, "avx512 int16");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_int16_load_store)
{
    std::ofstream out("log/avx512_int16_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int16_t, 32, 64>(out, "avx512 int16");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_uint16_basic)
{
    std::ofstream out("log/avx512_uint16_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint16_t, 32, 64>(out, "avx512 uint16");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_uint16_load_store)
{
    std::ofstream out("log/avx512_uint16_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint16_t, 32, 64>(out, "avx512 uint16");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_int32_basic)
{
    std::ofstream out("log/avx512_int32_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int32_t, 16, 64>(out, "avx512 int32");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_int32_load_store)
{
    std::ofstream out("log/avx512_int32_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int32_t, 16, 64>(out, "avx512 int32");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_uint32_basic)
{
    std::ofstream out("log/avx512_uint32_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint32_t, 16, 64>(out, "avx512 uint32");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_uint32_load_store)
{
    std::ofstream out("log/avx512_uint32_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint32_t, 16, 64>(out, "avx512 uint32");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_int64_basic)
{
    std::ofstream out("log/avx512_int64_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int64_t, 8, 64>(out, "avx512 int64");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_int64_load_store)
{
    std::ofstream out("log/avx512_int64_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int64_t, 8, 64>(out, "avx512 int64");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_uint64_basic)
{
    std::ofstream out("log/avx512_uint64_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint64_t, 8, 64>(out, "avx512 uint64");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_uint64_load_store)
{
    std::ofstream out("log/avx512_uint64_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint64_t, 8, 64>(out, "avx512 uint64");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_complex_float_basic)
{
    std::ofstream out("log/avx512_complex_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<std::complex<float>, 16, 64>(out, "avx512 complex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_complex_double_basic)
{
    std::ofstream out("log/avx512_complex_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<std::complex<double>, 8, 64>(out, "avx512 complex double");
    EXPECT_TRUE(res);
}

#ifdef XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, avx512_xtl_xcomplex_float_basic)
{
    std::ofstream out("log/avx512_xtl_xcomplex_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<xtl::xcomplex<float>, 16, 64>(out, "avx512 xtl xcomplex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_xtl_xcomplex_double_basic)
{
    std::ofstream out("log/avx512_xtl_xcomplex_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<xtl::xcomplex<double>, 8, 64>(out, "avx512 xtl xcomplex double");
    EXPECT_TRUE(res);
}
#endif

TEST(xsimd, avx512_conversion)
{
    std::ofstream out("log/avx512_conversion.log", std::ios_base::out);
    bool res = xsimd::test_simd_convert<8, 64>(out, "avx512 conversion");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_batch_cast)
{
    std::ofstream out("log/avx512_batch_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_batch_cast<8, 64>(out, "avx512 batch cast");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_bitwise_cast)
{
    std::ofstream out("log/avx512_bitwise_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_bitwise_cast<8, 64>(out, "avx512 bitwise cast");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_complex_float_load_store)
{
    std::ofstream out("log/avx512_complex_float_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<std::complex<float>, 16, 64>(out, "avx512 complex float load store");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_complex_double_load_store)
{
    std::ofstream out("log/avx512_complex_double_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<std::complex<double>, 8, 64>(out, "avx512 complex float double store");
    EXPECT_TRUE(res);
}

#ifdef XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, avx512_xtl_xcomplex_float_load_store)
{
    std::ofstream out("log/avx512_xtl_xcomplex_float_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<xtl::xcomplex<float>, 16, 64>(out, "avx512 xtl xcomplex float load store");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_xtl_xcomplex_double_load_store)
{
    std::ofstream out("log/avx512_xtl_xcomplex_double_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<xtl::xcomplex<double>, 8, 64>(out, "avx512 xtl xcomplex double load store");
    EXPECT_TRUE(res);
}
#endif
#endif

/********************
 * neon basic tests *
 ********************/

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM7_NEON_VERSION
TEST(xsimd, neon_float_basic)
{
    std::ofstream out("log/neon_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<float, 4, 32>(out, "neon float");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_int8_basic)
{
    std::ofstream out("log/neon_int8_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int8_t, 16, 32>(out, "neon int8");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_int8_load_store)
{
    std::ofstream out("log/neon_int8_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int8_t, 16, 32>(out, "neon int8");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_uint8_basic)
{
    std::ofstream out("log/neon_uint8_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint8_t, 16, 32>(out, "neon uint8");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_uint8_load_store)
{
    std::ofstream out("log/neon_uint8_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint8_t, 16, 32>(out, "neon uint8");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_int16_basic)
{
    std::ofstream out("log/neon_int16_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int16_t, 8, 32>(out, "neon int16");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_int16_load_store)
{
    std::ofstream out("log/neon_int16_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int16_t, 8, 32>(out, "neon int16");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_uint16_basic)
{
    std::ofstream out("log/neon_uint16_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint16_t, 8, 32>(out, "neon uint16");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_uint16_load_store)
{
    std::ofstream out("log/neon_uint16_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint16_t, 8, 32>(out, "neon uint16");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_int32_basic)
{
    std::ofstream out("log/neon_int32_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int32_t, 4, 32>(out, "neon int32");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_int32_load_store)
{
    std::ofstream out("log/neon_int32_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int32_t, 4, 32>(out, "neon int32");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_uint32_basic)
{
    std::ofstream out("log/neon_uint32_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint32_t, 4, 32>(out, "neon uint32");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_uint32_load_store)
{
    std::ofstream out("log/neon_uint32_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint32_t, 4, 32>(out, "neon uint32");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_int64_basic)
{
    std::ofstream out("log/neon_int64_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int64_t, 2, 32>(out, "neon int64");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_int64_load_store)
{
    std::ofstream out("log/neon_int64_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<int64_t, 2, 32>(out, "neon int64");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_uint64_basic)
{
    std::ofstream out("log/neon_uint64_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint64_t, 2, 32>(out, "neon uint64");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_uint64_load_store)
{
    std::ofstream out("log/neon_uint64_load_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_load_store<uint64_t, 2, 32>(out, "neon uint64");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_complex_float_basic)
{
    std::ofstream out("log/neon_complex_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<std::complex<float>, 4, 32>(out, "neon complex float");
    EXPECT_TRUE(res);
}

#ifdef XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, neon_xtl_xcomplex_float_basic)
{
    std::ofstream out("log/neon_xtl_xcomplex_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<xtl::xcomplex<float>, 4, 16>(out, "neon xtl xcomplex float");
    EXPECT_TRUE(res);
}
#endif

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
TEST(xsimd, neon_double_basic)
{
    std::ofstream out("log/neon_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<double, 2, 32>(out, "neon double");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_complex_double_basic)
{
    std::ofstream out("log/neon_complex_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<std::complex<double>, 2, 32>(out, "neon complex double");
    EXPECT_TRUE(res);
}

#ifdef XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, neon_xtl_xcomplex_double_basic)
{
    std::ofstream out("log/sse_xtl_xcomplex_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<xtl::xcomplex<double>, 2, 16>(out, "sse xtl xcomplex double");
    EXPECT_TRUE(res);
}
#endif

TEST(xsimd, neon_conversion)
{
    std::ofstream out("log/neon_conversion.log", std::ios_base::out);
    bool res = xsimd::test_simd_convert<2, 16>(out, "neon conversion");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_batch_cast)
{
    std::ofstream out("log/neon_batch_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_batch_cast<2, 16>(out, "neon batch cast");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_bitwise_cast)
{
    std::ofstream out("log/neon_bitwise_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_bitwise_cast<2, 16>(out, "neon bitwise cast");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_complex_float_load_store)
{
    std::ofstream out("log/neon_complex_float_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<std::complex<float>, 4, 16>(out, "neon complex float load store");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_complex_double_load_store)
{
    std::ofstream out("log/neon_complex_double_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<std::complex<double>, 2, 16>(out, "neon complex float double store");
    EXPECT_TRUE(res);
}

#ifdef XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, neon_xtl_xcomplex_float_load_store)
{
    std::ofstream out("log/neon_xtl_xcomplex_float_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<xtl::xcomplex<float>, 4, 16>(out, "neon xtl xcomplex float load store");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_xtl_xcomplex_double_load_store)
{
    std::ofstream out("log/neon_xtl_xcomplex_double_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<xtl::xcomplex<double>, 2, 16>(out, "neon xtl xcomplex double load store");
    EXPECT_TRUE(res);
}
#endif

#endif
#endif

/************************
 * fallback basic tests *
 ************************/

#if defined(XSIMD_ENABLE_FALLBACK)
TEST(xsimd, fallback_float_basic)
{
    std::ofstream out("log/fallback_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<float, 7, 32>(out, "fallback float");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_int32_basic)
{
    std::ofstream out("log/fallback_int32_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int32_t, 7, 32>(out, "fallback int32");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_int64_basic)
{
    std::ofstream out("log/fallback_int64_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int64_t, 3, 32>(out, "fallback int64");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_double_basic)
{
    std::ofstream out("log/fallback_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<double, 3, 32>(out, "fallback double");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_complex_float_basic)
{
    std::ofstream out("log/fallback_complex_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<std::complex<float>, 7, 32>(out, "fallback complex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_complex_double_basic)
{
    std::ofstream out("log/fallback_complex_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<std::complex<double>, 3, 32>(out, "fallback complex double");
    EXPECT_TRUE(res);
}

#ifdef XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, fallback_xtl_xcomplex_float_basic)
{
    std::ofstream out("log/fallback_xtl_xcomplex_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<xtl::xcomplex<float>, 7, 32>(out, "fallback xtl xcomplex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_xtl_xcomplex_double_basic)
{
    std::ofstream out("log/fallback_xtl_xcomplex_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_complex<xtl::xcomplex<double>, 3, 32>(out, "fallback xtl xcomplex double");
    EXPECT_TRUE(res);
}
#endif

TEST(xsimd, fallback_conversion)
{
    std::ofstream out("log/fallback_conversion.log", std::ios_base::out);
    bool res = xsimd::test_simd_convert<3, 32>(out, "fallback conversion");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_bitwise_cast)
{
    std::ofstream out("log/fallback_bitwise_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_bitwise_cast<3, 32>(out, "fallback bitwise cast");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_complex_float_load_store)
{
    std::ofstream out("log/fallback_complex_float_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<std::complex<float>, 7, 32>(out, "fallback complex float load store");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_complex_double_load_store)
{
    std::ofstream out("log/fallback_complex_double_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<std::complex<double>, 3, 32>(out, "fallback complex float double store");
    EXPECT_TRUE(res);
}

#ifdef XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, fallback_xtl_xcomplex_float_load_store)
{
    std::ofstream out("log/fallback_xtl_xcomplex_float_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<xtl::xcomplex<float>, 7, 32>(out, "fallback xtl xcomplex float load store");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_xtl_xcomplex_double_load_store)
{
    std::ofstream out("log/fallback_xtl_xcomplex_double_load_store.log", std::ios_base::out);
    bool res = xsimd::test_complex_simd_load_store<xtl::xcomplex<double>, 3, 32>(out, "fallback xtl xcomplex double load store");
    EXPECT_TRUE(res);
}
#endif
#endif
