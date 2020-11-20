/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <cstddef>
#include <vector>
#include <numeric>

#include "gtest/gtest.h"

#include "xsimd/config/xsimd_instruction_set.hpp"

#ifdef XSIMD_INSTR_SET_AVAILABLE

#include "xsimd/xsimd.hpp"

namespace xsimd
{
    struct interface_tester
    {
        std::vector<float, aligned_allocator<float, 64>> fvec;
        std::vector<int32_t, aligned_allocator<int32_t, 64>> ivec;
        std::vector<float, aligned_allocator<float, 64>> fres;
        std::vector<int32_t, aligned_allocator<int32_t, 64>> ires;

        interface_tester();

        static const std::size_t SIZE = simd_traits<float>::size;
    };

    interface_tester::interface_tester()
        : fvec(SIZE), ivec(SIZE), fres(SIZE), ires(SIZE)
    {
        std::iota(fvec.begin(), fvec.end(), 1.f);
        std::iota(ivec.begin(), ivec.end(), 1);
    }

    TEST(xsimd, set_simd)
    {
        interface_tester t;
        simd_type<float> r1 = set_simd(t.fvec[0]);
        EXPECT_EQ(r1[0], t.fvec[0]);

        simd_type<float> r2 = set_simd<int32_t, float>(t.ivec[0]);
        EXPECT_EQ(r2[0], t.fvec[0]);
    }

    TEST(xsimd, load_store_aligned)
    {
        interface_tester t;
        simd_type<float> r1 = load_aligned(&t.fvec[0]);
        store_aligned(&t.fres[0], r1);
        EXPECT_EQ(t.fvec, t.fres);

        simd_type<float> r2 = load_aligned<int32_t, float>(&t.ivec[0]);
        store_aligned(&t.fres[0], r2);
        EXPECT_EQ(t.fvec, t.fres);

        simd_type<float> r3 = load_aligned(&t.fvec[0]);
        store_aligned<int32_t, float>(&t.ires[0], r3);
        EXPECT_EQ(t.ivec, t.ires);
    }

    TEST(xsimd, load_store_unaligned)
    {
        interface_tester t;
        simd_type<float> r1 = load_unaligned(&t.fvec[0]);
        store_unaligned(&t.fres[0], r1);
        EXPECT_EQ(t.fvec, t.fres);

        simd_type<float> r2 = load_unaligned<int32_t, float>(&t.ivec[0]);
        store_unaligned(&t.fres[0], r2);
        EXPECT_EQ(t.fvec, t.fres);

        simd_type<float> r3 = load_unaligned(&t.fvec[0]);
        store_unaligned<int32_t, float>(&t.ires[0], r3);
        EXPECT_EQ(t.ivec, t.ires);
    }

    TEST(xsimd, load_store_simd_aligned)
    {
        interface_tester t;
        simd_type<float> r1 = load_simd(&t.fvec[0], aligned_mode());
        store_simd(&t.fres[0], r1, aligned_mode());
        EXPECT_EQ(t.fvec, t.fres);

        simd_type<float> r2 = load_simd<int32_t, float>(&t.ivec[0], aligned_mode());
        store_simd(&t.fres[0], r2, aligned_mode());
        EXPECT_EQ(t.fvec, t.fres);

        simd_type<float> r3 = load_simd(&t.fvec[0], aligned_mode());
        store_simd<int32_t, float>(&t.ires[0], r3, aligned_mode());
        EXPECT_EQ(t.ivec, t.ires);
    }

    TEST(xsimd, load_store_simd_unaligned)
    {
        interface_tester t;
        simd_type<float> r1 = load_simd(&t.fvec[0], unaligned_mode());
        store_simd(&t.fres[0], r1, unaligned_mode());
        EXPECT_EQ(t.fvec, t.fres);

        simd_type<float> r2 = load_simd<int32_t, float>(&t.ivec[0], unaligned_mode());
        store_simd(&t.fres[0], r2, unaligned_mode());
        EXPECT_EQ(t.fvec, t.fres);

        simd_type<float> r3 = load_simd(&t.fvec[0], unaligned_mode());
        store_simd<int32_t, float>(&t.ires[0], r3, unaligned_mode());
        EXPECT_EQ(t.ivec, t.ires);
    }
}
#endif // XSIMD_INSTR_SET_AVAILABLE
