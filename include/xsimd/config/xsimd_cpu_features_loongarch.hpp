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

#ifndef XSIMD_CPU_FEATURES_LOONGARCH_HPP
#define XSIMD_CPU_FEATURES_LOONGARCH_HPP

#include "./xsimd_config.hpp"
#include "./xsimd_getauxval.hpp"

#if XSIMD_TARGET_LOONGARCH64 && XSIMD_HAVE_LINUX_GETAUXVAL
// HWCAP_XXX masks to use on getauxval results.
// Header does not exists on all architectures and masks are architecture
// specific.
#include <asm/hwcap.h>
#endif // XSIMD_TARGET_LOONGARCH64 && XSIMD_HAVE_LINUX_GETAUXVAL

namespace xsimd
{
    /**
     * An opinionated CPU feature detection utility for LoongArch.
     *
     * On Linux, runtime detection uses getauxval to query the auxiliary vector.
     * On other platforms, only compile-time information is used.
     *
     * This is well defined on all architectures.
     * It will always return false on non-LoongArch architectures.
     */
    class loongarch_cpu_features : private linux_hwcap_backend_default
    {
    public:
        inline bool lsx() const noexcept;
        inline bool lasx() const noexcept;
    };

    /********************
     *  Implementation  *
     ********************/

    inline bool loongarch_cpu_features::lsx() const noexcept
    {
#if XSIMD_TARGET_LOONGARCH64 && XSIMD_HAVE_LINUX_GETAUXVAL
#ifdef HWCAP_LOONGARCH_LSX
        constexpr unsigned long loongarch_hwcap_lsx = HWCAP_LOONGARCH_LSX;
#else
        // Possibly missing on older Linux distributions
        constexpr unsigned long loongarch_hwcap_lsx = 1ul << 4;
#endif
        return hwcap().has_feature(loongarch_hwcap_lsx);
#else
        return XSIMD_WITH_LSX;
#endif
    }

    inline bool loongarch_cpu_features::lasx() const noexcept
    {
#if XSIMD_TARGET_LOONGARCH64 && XSIMD_HAVE_LINUX_GETAUXVAL
#ifdef HWCAP_LOONGARCH_LASX
        constexpr unsigned long loongarch_hwcap_lasx = HWCAP_LOONGARCH_LASX;
#else
        // Possibly missing on older Linux distributions
        constexpr unsigned long loongarch_hwcap_lasx = 1ul << 5;
#endif
        return hwcap().has_feature(loongarch_hwcap_lasx);
#else
        return XSIMD_WITH_LASX;
#endif
    }
}

#endif
