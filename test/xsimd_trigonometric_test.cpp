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

#include "xsimd/config/xsimd_instruction_set.hpp"

#ifdef XSIMD_INSTR_SET_AVAILABLE

#include "xsimd/math/xsimd_trigonometric.hpp"
#include "xsimd/memory/xsimd_aligned_allocator.hpp"
#include "xsimd/types/xsimd_types_include.hpp"
#include "xsimd_trigonometric_test.hpp"

namespace xsimd
{
    template <class T, size_t N, size_t A>
    bool test_trigonometric(std::ostream& out, const std::string& name)
    {
        simd_trigonometric_tester<T, N, A> tester(name);
        return test_simd_trigonometric(out, tester);
    }
}

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
TEST(xsimd, sse_float_trigonometric)
{
    std::ofstream out("log/sse_float_trigonometric.log", std::ios_base::out);
    bool res = xsimd::test_trigonometric<float, 4, 16>(out, "sse float");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_double_trigonometric)
{
    std::ofstream out("log/sse_double_trigonometric.log", std::ios_base::out);
    bool res = xsimd::test_trigonometric<double, 2, 16>(out, "sse double");
    EXPECT_TRUE(res);
}
#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
TEST(xsimd, avx_float_trigonometric)
{
    std::ofstream out("log/avx_float_trigonometric.log", std::ios_base::out);
    bool res = xsimd::test_trigonometric<float, 8, 32>(out, "avx float");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_double_trigonometric)
{
    std::ofstream out("log/avx_double_trigonometric.log", std::ios_base::out);
    bool res = xsimd::test_trigonometric<double, 4, 32>(out, "avx double");
    EXPECT_TRUE(res);
}
#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512_VERSION
TEST(xsimd, avx512_float_trigonometric)
{
    std::ofstream out("log/avx512_float_trigonometric.log", std::ios_base::out);
    bool res = xsimd::test_trigonometric<float, 16, 64>(out, "avx512 float");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx512_double_trigonometric)
{
    std::ofstream out("log/avx512_double_trigonometric.log", std::ios_base::out);
    bool res = xsimd::test_trigonometric<double, 8, 64>(out, "avx512 double");
    EXPECT_TRUE(res);
}
#endif

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM7_NEON_VERSION
TEST(xsimd, neon_float_trigonometric)
{
    std::ofstream out("log/neon_float_trigonometric.log", std::ios_base::out);
    bool res = xsimd::test_trigonometric<float, 4, 16>(out, "neon float");
    EXPECT_TRUE(res);
}
#endif
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
TEST(xsimd, neon_double_trigonometric)
{
    std::ofstream out("log/neon_double_trigonometric.log", std::ios_base::out);
    bool res = xsimd::test_trigonometric<double, 2, 32>(out, "neon double");
    EXPECT_TRUE(res);
}
#endif

#if defined(XSIMD_ENABLE_FALLBACK)
TEST(xsimd, fallback_float_trigonometric)
{
    std::ofstream out("log/fallback_float_trigonometric.log", std::ios_base::out);
    bool res = xsimd::test_trigonometric<float, 7, 32>(out, "fallback float");
    EXPECT_TRUE(res);
}

TEST(xsimd, fallback_double_trigonometric)
{
    std::ofstream out("log/fallback_double_trigonometric.log", std::ios_base::out);
    bool res = xsimd::test_trigonometric<double, 3, 32>(out, "fallback double");
    EXPECT_TRUE(res);
}
#endif
#endif // XSIMD_INSTR_SET_AVAILABLE
