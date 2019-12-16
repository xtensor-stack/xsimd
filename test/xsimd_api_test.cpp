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
#include "xsimd_api_test.hpp"

namespace xsimd
{
    template <size_t N, size_t A>
    bool test_api_load(std::ostream& out, const std::string& name)
    {
        simd_api_load_store_tester<N, A> tester(name);
        return test_simd_api_load(out, tester);
    }

    template <size_t N, size_t A>
    bool test_api_store(std::ostream& out, const std::string& name)
    {
        simd_api_load_store_tester<N, A> tester(name);
        return test_simd_api_store(out, tester);
    }

    template <class T, size_t N, size_t A>
    bool test_complex_api(std::ostream& out, const std::string& name)
    {
        simd_complex_ls_tester<T, N, A> tester(name);
        return test_simd_complex_api(out, tester);
    }
}

#if XSIMD_X86_INSTR_SET > XSIMD_VERSION_NUMBER_NOT_AVAILABLE
TEST(xsimd, complex_return_type)
{
    using cf_type = std::complex<float>;
    using cf_return_type = xsimd::simd_return_type<float, cf_type>;
    EXPECT_TRUE((std::is_same<cf_return_type, xsimd::batch<cf_type, XSIMD_BATCH_FLOAT_SIZE>>::value));

    using df_type = std::complex<double>;
    using df_return_type = xsimd::simd_return_type<double, df_type>;
    EXPECT_TRUE((std::is_same<df_return_type, xsimd::batch<df_type, XSIMD_BATCH_DOUBLE_SIZE>>::value));
}

#ifdef XSIMD_ENABLE_XTL_COMPLEX
TEST(xsimd, xtl_xcomplex_return_type)
{
    using cf_type = xtl::xcomplex<float>;
    using cf_return_type = xsimd::simd_return_type<float, cf_type>;
    EXPECT_TRUE((std::is_same<cf_return_type, xsimd::batch<cf_type, XSIMD_BATCH_FLOAT_SIZE>>::value));

#ifdef XSIMD_BATCH_DOUBLE_SIZE
    using df_type = xtl::xcomplex<double>;
    using df_return_type = xsimd::simd_return_type<double, df_type>;
    EXPECT_TRUE((std::is_same<df_return_type, xsimd::batch<df_type, XSIMD_BATCH_DOUBLE_SIZE>>::value));
#endif
}
#endif

TEST(xsimd, api_load)
{
    std::ofstream out("log/xsimd_api_load.log", std::ios_base::out);
    bool res = xsimd::test_api_load<XSIMD_BATCH_DOUBLE_SIZE, XSIMD_DEFAULT_ALIGNMENT>(out, "xsimd api load");
    EXPECT_TRUE(res);
}

TEST(xsimd, api_store)
{
    std::ofstream out("log/xsimd_api_store.log", std::ios_base::out);
    bool res = xsimd::test_api_store<XSIMD_BATCH_DOUBLE_SIZE, XSIMD_DEFAULT_ALIGNMENT>(out, "xsimd api store");
    EXPECT_TRUE(res);
}

TEST(xsimd, complex_float_api)
{
    std::ofstream out("log/xsimd_complex_float_api.log", std::ios_base::out);
    bool res = xsimd::test_complex_api<std::complex<float>, XSIMD_BATCH_FLOAT_SIZE, XSIMD_DEFAULT_ALIGNMENT>(out, "xsimd complex float api");
    EXPECT_TRUE(res);
}

TEST(xsimd, complex_double_api)
{
    std::ofstream out("log/xsimd_complex_double_api.log", std::ios_base::out);
    bool res = xsimd::test_complex_api<std::complex<double>, XSIMD_BATCH_DOUBLE_SIZE, XSIMD_DEFAULT_ALIGNMENT>(out, "xsimd complex double api");
    EXPECT_TRUE(res);
}

#ifdef XSIMD_BATCH_DOUBLE_SIZE
TEST(xsimd, api_set)
{
    std::ofstream out("log/xsimd_set_api.log", std::ios_base::out);
    bool res = xsimd::test_simd_api_set(out, "xsimd api set");
    EXPECT_TRUE(res);
}
#endif

#endif
