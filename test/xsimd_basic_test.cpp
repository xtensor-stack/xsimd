/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
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

#include "xsimd/memory/xsimd_aligned_allocator.hpp"
#include "xsimd/types/xsimd_types_include.hpp"

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
    bool test_simd_cast(std::ostream& out, const std::string& name)
    {
        simd_cast_tester<N, A> tester(name);
        return test_simd_cast(out, tester);
    }

    template <size_t N, size_t A>
    bool test_simd_load(std::ostream& out, const std::string& name)
    {
        simd_load_store_tester<N, A> tester(name);
        return test_simd_load(out, tester);
    }

    template <size_t N, size_t A>
    bool test_simd_store(std::ostream& out, const std::string& name)
    {
        simd_load_store_tester<N, A> tester(name);
        return test_simd_store(out, tester);
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

TEST(xsimd, sse_uint8_basic)
{
    std::ofstream out("log/sse_uint8_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint8_t, 16, 16>(out, "sse uint8");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_int32_basic)
{
    std::ofstream out("log/sse_int32_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int32_t, 4, 16>(out, "sse int32");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_int64_basic)
{
    std::ofstream out("log/sse_int64_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int64_t, 2, 16>(out, "sse int64");
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

TEST(xsimd, sse_cast)
{
    std::ofstream out("log/sse_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_cast<2, 16>(out, "sse cast");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_load)
{
    std::ofstream out("log/sse_load.log", std::ios_base::out);
    bool res = xsimd::test_simd_load<2, 16>(out, "sse load");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_store)
{
    std::ofstream out("log/sse_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_store<2, 16>(out, "sse store");
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

TEST(xsimd, avx_uint8_basic)
{
    std::ofstream out("log/avx_uint8_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint8_t, 32, 32>(out, "avx uint8");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_int32_basic)
{
    std::ofstream out("log/avx_int32_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int32_t, 8, 32>(out, "avx int32");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_int64_basic)
{
    std::ofstream out("log/avx_int64_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int64_t, 4, 32>(out, "avx int64");
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

TEST(xsimd, avx_cast)
{
    std::ofstream out("log/avx_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_cast<4, 32>(out, "avx cast");
    EXPECT_TRUE(res);
}
TEST(xsimd, avx_load)
{
    std::ofstream out("log/avx_load.log", std::ios_base::out);
    bool res = xsimd::test_simd_load<4, 32>(out, "avx load");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_store)
{
    std::ofstream out("log/avx_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_store<4, 32>(out, "avx store");
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

TEST(xsimd, avx512_uint8_basic)
{
    std::ofstream out("log/avx512_uint8_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint8_t, 64, 64>(out, "avx512 uint8");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_int32_basic)
{
    std::ofstream out("log/avx512_int32_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int32_t, 16, 64>(out, "avx512 int32");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_int64_basic)
{
    std::ofstream out("log/avx512_int64_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int64_t, 8, 64>(out, "avx512 int64");
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
    bool res = xsimd::test_simd_complex<xtl::xcomplex<double>, 16, 64>(out, "avx512 xtl xcomplex double");
    EXPECT_TRUE(res);
}
#endif

TEST(xsimd, avx512_conversion)
{
    std::ofstream out("log/avx512_conversion.log", std::ios_base::out);
    bool res = xsimd::test_simd_convert<8, 64>(out, "avx512 conversion");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_cast)
{
    std::ofstream out("log/avx512_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_cast<8, 64>(out, "avx512 cast");
    EXPECT_TRUE(res);
}
TEST(xsimd, avx512_load)
{
    std::ofstream out("log/avx512_load.log", std::ios_base::out);
    bool res = xsimd::test_simd_load<8, 64>(out, "avx512 load");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_store)
{
    std::ofstream out("log/avx512_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_store<8, 64>(out, "avx512 store");
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

TEST(xsimd, neon_uint8_basic)
{
    std::ofstream out("log/neon_uint8_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<uint8_t, 16, 32>(out, "neon uint8");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_int32_basic)
{
    std::ofstream out("log/neon_int32_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int32_t, 4, 32>(out, "neon int32");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_int64_basic)
{
    std::ofstream out("log/neon_int64_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<int64_t, 2, 32>(out, "neon int64");
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

TEST(xsimd, neon_cast)
{
    std::ofstream out("log/neon_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_cast<2, 16>(out, "neon cast");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_load)
{
    std::ofstream out("log/neon_load.log", std::ios_base::out);
    bool res = xsimd::test_simd_load<2, 16>(out, "neon load");
    EXPECT_TRUE(res);
}

TEST(xsimd, neon_store)
{
    std::ofstream out("log/neon_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_store<2, 16>(out, "neon store");
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

TEST(xsimd, fallback_cast)
{
    std::ofstream out("log/fallback_cast.log", std::ios_base::out);
    bool res = xsimd::test_simd_cast<3, 32>(out, "fallback cast");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_load)
{
    std::ofstream out("log/fallback_load.log", std::ios_base::out);
    bool res = xsimd::test_simd_load<3, 32>(out, "fallback load");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_store)
{
    std::ofstream out("log/fallback_store.log", std::ios_base::out);
    bool res = xsimd::test_simd_store<3, 32>(out, "fallback store");
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