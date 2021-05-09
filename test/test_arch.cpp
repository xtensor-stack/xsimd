/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <random>

#include "test_utils.hpp"

static_assert(xsimd::arch::default_::configured, "default arch must be configured");
static_assert(xsimd::arch::all::contains<xsimd::arch::default_>(), "default arch is a valid arch");
static_assert(xsimd::arch::configured::contains<xsimd::arch::default_>(), "default arch is configured");
static_assert(!(xsimd::arch::x86::configured & xsimd::arch::arm::configured), "either x86 or arm, but not both");
