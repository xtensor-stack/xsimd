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
#include "xsimd/types/xsimd_sse_double.hpp"
#include "xsimd/types/xsimd_sse_float.hpp"
#include "xsimd/types/xsimd_avx_double.hpp"
#include "xsimd/types/xsimd_avx_float.hpp"
#include "xsimd_common_test.hpp"

namespace xsimd
{
    template <class V, size_t N, size_t A>
    bool test_simd(std::ostream& out, const std::string& name)
    {
        simd_basic_tester<V, N, A> tester(name);
        return test_simd_common(out, tester);
    }
}

TEST(xsimd, sse_float_basic)
{
    std::ofstream out("log/sse_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<xsimd::vector4f, 4, 16>(out, "sse float");
    EXPECT_TRUE(res);
}

TEST(xsimd, sse_double_basic)
{
    std::ofstream out("log/sse_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<xsimd::vector2d, 2, 16>(out, "sse double");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_float_basic)
{
    std::ofstream out("log/avx_float_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<xsimd::vector8f, 8, 32>(out, "avx float");
    EXPECT_TRUE(res);
}

TEST(xsimd, avx_double_basic)
{
    std::ofstream out("log/avx_double_basic.log", std::ios_base::out);
    bool res = xsimd::test_simd<xsimd::vector4d, 4, 32>(out, "avx double");
    EXPECT_TRUE(res);
}

