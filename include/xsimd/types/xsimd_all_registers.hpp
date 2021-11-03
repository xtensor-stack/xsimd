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

#include "../types/xsimd_fma3_register.hpp"
#include "../types/xsimd_sse2_register.hpp"
#include "../types/xsimd_sse3_register.hpp"
#include "../types/xsimd_sse4_1_register.hpp"
#include "../types/xsimd_sse4_2_register.hpp"

#include "../types/xsimd_avx2_register.hpp"
#include "../types/xsimd_avx_register.hpp"
#include "../types/xsimd_fma5_register.hpp"

#include "../types/xsimd_avx512bw_register.hpp"
#include "../types/xsimd_avx512cd_register.hpp"
#include "../types/xsimd_avx512dq_register.hpp"
#include "../types/xsimd_avx512f_register.hpp"

#include "xsimd_neon64_register.hpp"
#include "xsimd_neon_register.hpp"
