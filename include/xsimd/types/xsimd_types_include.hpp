/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_TYPES_INCLUDE_HPP
#define XSIMD_TYPES_INCLUDE_HPP

#include "../config/xsimd_include.hpp"

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION
#include "xsimd_sse_conversion.hpp"
#include "xsimd_sse_double.hpp"
#include "xsimd_sse_float.hpp"
#include "xsimd_sse_int32.hpp"
#include "xsimd_sse_int64.hpp"
#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
#include "xsimd_avx_conversion.hpp"
#include "xsimd_avx_double.hpp"
#include "xsimd_avx_float.hpp"
#include "xsimd_avx_int32.hpp"
#include "xsimd_avx_int64.hpp"
#endif

#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM7_NEON_VERSION
#include "xsimd_neon_conversion.hpp"
#include "xsimd_neon_bool.hpp"
#if XSIMD_ARM_INSTR_SET >= XSIMD_ARM8_64_NEON_VERSION
    #include "xsimd_neon_double.hpp"
#endif
#include "xsimd_neon_float.hpp"
#include "xsimd_neon_int64.hpp"
#include "xsimd_neon_int32.hpp"
#endif

#include "xsimd_utils.hpp"

#endif
