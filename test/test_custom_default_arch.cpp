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
#ifdef __STSE2__

#define XSIMD_DEFAULT_ARCH xsimd::sse2
#include "xsimd/xsimd.hpp"

#include "test_utils.hpp"

// Could be different than sse2 if we compile for other architecture avx
static_assert(std::is_same<xsimd::default_arch, xsimd::sse2>::value, "default arch correctly hooked");

#else

#undef XSIMD_DEFAULT_ARCH
#define XSIMD_DEFAULT_ARCH xsimd::unsupported

#include "xsimd/xsimd.hpp"

#endif
