/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 * Copyright (c) Marco Barbone                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "./xsimd_arm_registers.hpp"
#include "./xsimd_rvv_register.hpp"
#include "./xsimd_vsx_register.hpp"
#include "./xsimd_vxe_register.hpp"
#include "./xsimd_wasm_register.hpp"
#include "./xsimd_x86_registers.hpp"

#if XSIMD_WITH_EMULATED
#include "./xsimd_emulated_register.hpp"
#endif
