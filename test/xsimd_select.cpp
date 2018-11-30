/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <fstream>
#include <iostream>

#include "xsimd/xsimd.hpp"

#include "gtest/gtest.h"

TEST(xsimd, compile_time_select)
{
    xsimd::batch<float, 8> a( 0,  1,  2,  3,  4,  5,  6,  7),
                           b(10, 20, 30, 40, 50, 60, 70, 80);

    auto r1 = xsimd::select<0b00110011>(a, b);
    EXPECT_TRUE(all(r1 == xsimd::batch<float, 8>(0, 1, 30, 40, 4, 5, 70, 80)));

    r1 = xsimd::select<0b11100011>(a, b);
    EXPECT_TRUE(all(r1 == xsimd::batch<float, 8>(10, 20, 30, 3, 4, 5, 70, 80)));

    xsimd::batch<double, 4> c(0,1,2,3),
                            d(10, 20, 30, 40);

    auto r2 = xsimd::select<0b1110>(c, d);
    EXPECT_TRUE(all(r2 == xsimd::batch<double, 4>(10, 20, 30, 3)));
}