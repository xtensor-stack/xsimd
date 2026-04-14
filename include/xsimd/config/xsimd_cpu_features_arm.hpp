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

#ifndef XSIMD_CPU_FEATURES_ARM_HPP
#define XSIMD_CPU_FEATURES_ARM_HPP

#include "./xsimd_config.hpp"
#include "./xsimd_getauxval.hpp"

#if XSIMD_TARGET_ARM && XSIMD_HAVE_LINUX_GETAUXVAL
// HWCAP_XXX masks to use on getauxval results.
// Header does not exists on all architectures and masks are architecture
// specific.
#include <asm/hwcap.h>
#endif // XSIMD_TARGET_ARM && XSIMD_HAVE_LINUX_GETAUXVAL

namespace xsimd
{
    /**
     * An opinionated CPU feature detection utility for ARM.
     *
     * Combines compile-time knowledge with runtime detection when available.
     * On Linux, runtime detection uses getauxval to query the auxiliary vector.
     * On other platforms, only compile-time information is used.
     *
     * This is well defined on all architectures.
     * It will always return false on non-ARM architectures.
     */
    class arm_cpu_features : private linux_hwcap_backend_default
    {
    public:
        inline bool neon() const noexcept;
        inline bool neon64() const noexcept;
        inline bool sve() const noexcept;
        inline bool i8mm() const noexcept;
    };

    /********************
     *  Implementation  *
     ********************/

    inline bool arm_cpu_features::neon() const noexcept
    {
#if XSIMD_TARGET_ARM && !XSIMD_TARGET_ARM64 && XSIMD_HAVE_LINUX_GETAUXVAL
        return hwcap().has_feature(HWCAP_NEON);
#else
        return static_cast<bool>(XSIMD_WITH_NEON);
#endif
    }

    inline bool arm_cpu_features::neon64() const noexcept
    {
        return static_cast<bool>(XSIMD_WITH_NEON64);
    }

    inline bool arm_cpu_features::sve() const noexcept
    {
#if XSIMD_TARGET_ARM64 && XSIMD_HAVE_LINUX_GETAUXVAL
        return hwcap().has_feature(HWCAP_SVE);
#else
        return false;
#endif
    }

    inline bool arm_cpu_features::i8mm() const noexcept
    {
#if XSIMD_TARGET_ARM64 && XSIMD_HAVE_LINUX_GETAUXVAL
#ifdef HWCAP2_I8MM
        return hwcap2().has_feature(HWCAP2_I8MM);
#else
        // Possibly missing on older Linux distributions
        return hwcap2().has_feature(1 << 13);
#endif
#else
        return false;
#endif
    }
}
#endif
