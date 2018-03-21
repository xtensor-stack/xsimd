/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <fstream>
#include <iostream>
#include <map>

#include "gtest/gtest.h"

#include "xsimd/memory/xsimd_aligned_allocator.hpp"
#include "xsimd/types/xsimd_types_include.hpp"

#include "xsimd_basic_test.hpp"

namespace xsimd
{
    template <class T, size_t N, size_t A>
    bool test_simd(std::ostream& out, const std::string& name)
    {
        simd_basic_tester<T, N, A> tester(name);
        return test_simd_basic(out, tester);
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
}

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
#endif

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
#endif

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM7_NEON_VERSION
TEST(xsimd, neon_float_basic)
{
    std::ofstream out("log/neon_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<float, 4, 16>(out, "neon float");
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

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
TEST(xsimd, neon_double_basic)
{
    std::ofstream out("log/neon_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<double, 2, 32>(out, "neon double");
    EXPECT_TRUE(res);
}

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
#endif
#endif