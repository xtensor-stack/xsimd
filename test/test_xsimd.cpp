/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <iostream>
#include <fstream>
#include <map>

#include "gtest/gtest.h"

#include "xsimd/config/xsimd_include.hpp"
#include "xsimd/memory/xsimd_aligned_allocator.hpp"

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
#include "xsimd/types/xsimd_sse_double.hpp"
#include "xsimd/types/xsimd_sse_float.hpp"
#include "xsimd/types/xsimd_sse_int.hpp"
#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
#include "xsimd/types/xsimd_avx_double.hpp"
#include "xsimd/types/xsimd_avx_float.hpp"
#include "xsimd/types/xsimd_avx_int.hpp"
#endif

#include "xsimd_common_test.hpp"

namespace xsimd
{
    template <class V, size_t N, size_t A>
    bool test_simd(std::ostream& out, const std::string& name)
    {
        simd_basic_tester<V, N, A> tester(name);
        return test_simd_common(out, tester);
    }

    template <class V, size_t N, size_t A>
    bool test_simd_int(std::ostream& out, const std::string& name)
    {
        simd_basic_int_tester<V, N, A> tester(name);
        return test_simd_common_int(out, tester);
    }
}

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
TEST(xsimd, sse_float_basic)
{
    std::ofstream out("log/sse_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<xsimd::batch<float, 4>, 4, 16>(out, "sse float");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_double_basic)
{
    std::ofstream out("log/sse_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<xsimd::batch<double, 2>, 2, 16>(out, "sse double");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_int_basic)
{
    std::ofstream out("log/sse_int_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<xsimd::batch<int, 4>, 4, 16>(out, "sse int");
    EXPECT_TRUE(res);
}
#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
TEST(xsimd, avx_float_basic)
{
    std::ofstream out("log/avx_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<xsimd::batch<float, 8>, 8, 32>(out, "avx float");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_double_basic)
{
    std::ofstream out("log/avx_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<xsimd::batch<double, 4>, 4, 32>(out, "avx double");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_int_basic)
{
    std::ofstream out("log/sse_avx_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd_int<xsimd::batch<int, 8>, 8, 32>(out, "avx int");
    EXPECT_TRUE(res);
}
#endif
