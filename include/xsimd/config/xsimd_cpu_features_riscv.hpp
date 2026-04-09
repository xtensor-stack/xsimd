/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ***************************************************************************/

#ifndef XSIMD_CPU_FEATURES_RISCV_HPP
#define XSIMD_CPU_FEATURES_RISCV_HPP

#include "./xsimd_config.hpp"
#include "./xsimd_getauxval.hpp"

#if XSIMD_TARGET_RISCV && XSIMD_HAVE_LINUX_GETAUXVAL
// HWCAP_XXX masks to use on getauxval results.
// Header does not exists on all architectures and masks are architecture
// specific.
#include <asm/hwcap.h>
#endif // XSIMD_TARGET_RISCV && XSIMD_HAVE_LINUX_GETAUXVAL

namespace xsimd
{
    class riscv_cpu_features : private linux_hwcap_backend_default
    {
    public:
        inline bool rvv() const noexcept;
    };

    /********************
     *  Implementation  *
     ********************/

    inline bool riscv_cpu_features::rvv() const noexcept
    {
#if XSIMD_TARGET_RISCV && XSIMD_HAVE_LINUX_GETAUXVAL
#ifdef HWCAP_V
        return hwcap().has_feature(HWCAP_V);
#else
        // Possibly missing on older Linux distributions
        return hwcap().has_feature(1 << ('V' - 'A'));
#endif
#else
        return false;
#endif
    }
}

#endif
