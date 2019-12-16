/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <fstream>
#include <iostream>

#include "gtest/gtest.h"

#include "xsimd/math/xsimd_math_complex.hpp"
#include "xsimd/memory/xsimd_aligned_allocator.hpp"
#include "xsimd/types/xsimd_types_include.hpp"
#include "xsimd_cpower_test.hpp"

namespace xsimd
{
    template <class T, size_t N, size_t A>
    bool test_cpower(std::ostream& out, const std::string& name)
    {
        simd_cpower_tester<T, N, A> tester(name);
        return test_simd_cpower(out, tester);
    }
}

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
TEST(xsimd, sse_complex_float_power)
{
    std::ofstream out("log/sse_complex_float_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<std::complex<float>, 4, 16>(out, "sse complex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_complex_double_power)
{
    std::ofstream out("log/sse_complex_double_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<std::complex<double>, 2, 16>(out, "sse complex double");
    EXPECT_TRUE(res);
}

#if XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, sse_xtl_xcomplex_float_power)
{
    std::ofstream out("log/sse_xtl_xcomplex_float_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<xtl::xcomplex<float>, 4, 16>(out, "sse xtl xcomplex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_xtl_xcomplex_double_power)
{
    std::ofstream out("log/sse_xtl_xcomplex_double_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<xtl::xcomplex<double>, 2, 16>(out, "sse xtl xcomplex double");
    EXPECT_TRUE(res);
}
#endif

#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
TEST(xsimd, avx_complex_float_power)
{
    std::ofstream out("log/avx_complex_float_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<std::complex<float>, 8, 32>(out, "avx complex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_complex_double_power)
{
    std::ofstream out("log/avx_complex_double_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<std::complex<double>, 4, 32>(out, "avx complex double");
    EXPECT_TRUE(res);
}

#if XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, avx_xtl_xcomplex_float_power)
{
    std::ofstream out("log/avx_xtl_xcomplex_float_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<xtl::xcomplex<float>, 8, 32>(out, "avx xtl xcomplex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_xtl_xcomplex_double_power)
{
    std::ofstream out("log/avx_xtl_xcomplex_double_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<xtl::xcomplex<double>, 4, 32>(out, "avx xtl xcomplex double");
    EXPECT_TRUE(res);
}
#endif

#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512_VERSION
#if DEBUG_FLOAT_ACCURACY
TEST(xsimd, avx512_complex_float_power)
{
    std::ofstream out("log/avx512_complex_float_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<std::complex<float>, 16, 64>(out, "avx512 complex float");
    EXPECT_TRUE(res);
}
#endif

TEST(xsimd, avx512_complex_double_power)
{
    std::ofstream out("log/avx512_complex_double_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<std::complex<double>, 8, 64>(out, "avx512 complex double");
    EXPECT_TRUE(res);
}

#if XSIMD_ENABLE_XTL_COMPLEX
#if DEBUG_FLOAT_ACCURACY
TEST(xsimd, avx512_xtl_xcomplex_float_power)
{
    std::ofstream out("log/avx512_xtl_xcomplex_float_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<xtl::xcomplex<float>, 16, 64>(out, "avx512 xtl xcomplex float");
    EXPECT_TRUE(res);
}
#endif

TEST(xsimd, avx512_xtl_xcomplex_double_power)
{
    std::ofstream out("log/avx512_xtl_xcomplex_double_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<xtl::xcomplex<double>, 8, 64>(out, "avx512 xtl xcomplex double");
    EXPECT_TRUE(res);
}
#endif

#endif

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM7_NEON_VERSION
#if DEBUG_FLOAT_ACCURACY
TEST(xsimd, neon_complex_float_power)
{
    std::ofstream out("log/neon_complex_float_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<std::complex<float>, 4, 16>(out, "neon complex float");
    EXPECT_TRUE(res);
}
#if XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, neon_xtl_xcomplex_float_power)
{
    std::ofstream out("log/neon_xtl_xcomplex_float_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<xtl::xcomplex<float>, 4, 16>(out, "neon xtl xcomplex float");
    EXPECT_TRUE(res);
}
#endif
#endif
#endif
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
TEST(xsimd, neon_complex_double_power)
{
    std::ofstream out("log/neon_complex_double_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<std::complex<double>, 2, 32>(out, "neon complex double");
    EXPECT_TRUE(res);
}
#if XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, neon_xtl_xcomplex_double_power)
{
    std::ofstream out("log/neon_xtl_xcomplex_double_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<xtl::xcomplex<double>, 2, 32>(out, "neon xtl xcomplex double");
    EXPECT_TRUE(res);
}
#endif
#endif

#if defined(XSIMD_ENABLE_FALLBACK)
TEST(xsimd, fallback_complex_float_power)
{
    std::ofstream out("log/fallback_complex_float_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<std::complex<float>, 7, 32>(out, "fallback complex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_complex_double_power)
{
    std::ofstream out("log/fallback_complex_double_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<std::complex<double>, 3, 32>(out, "fallback complex double");
    EXPECT_TRUE(res);
}

#if XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, fallback_xtl_xcomplex_float_power)
{
    std::ofstream out("log/fallback_xtl_xcomplex_float_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<xtl::xcomplex<float>, 7, 32>(out, "fallback xtl xcomplex float");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_xtl_xcomplex_double_power)
{
    std::ofstream out("log/fallback_xtl_xcomplex_double_power.log", std::ios_base::out);
    bool res = xsimd::test_cpower<xtl::xcomplex<double>, 3, 32>(out, "fallback xtl xcomplex double");
    EXPECT_TRUE(res);
}
#endif

#endif
