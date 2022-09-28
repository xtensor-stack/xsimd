/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

/*
 * Make sure the inclusion works correctly without _GNU_SOURCE
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "xsimd/xsimd.hpp"

#include "doctest/doctest.h"

TEST_CASE("[GNU_SOURCE support]")
{

    SUBCASE("exp10")
    {
        CHECK_EQ(xsimd::exp10(0.), 1.);
        CHECK_EQ(xsimd::exp10(0.f), 1.f);
    }
}
