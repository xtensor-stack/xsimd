/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "test_utils.hpp"
#include "xsimd/stl/algorithms.hpp"
#include <numeric>

#include <vector>

TEST(simd_traits, detail_has_batch)
{
    EXPECT_FALSE(xsimd::detail::has_batch<std::vector<std::size_t>>::value);
}

