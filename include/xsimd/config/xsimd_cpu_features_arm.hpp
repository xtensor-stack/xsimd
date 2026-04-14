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

#if XSIMD_TARGET_ARM && XSIMD_HAVE_LINUX_GETAUXVAL
#include "../utils/bits.hpp"
#include "./xsimd_getauxval.hpp"

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
    class arm_cpu_features
    {
    public:
        arm_cpu_features() noexcept = default;

        inline bool neon() const noexcept
        {
#if XSIMD_TARGET_ARM && !XSIMD_TARGET_ARM64 && XSIMD_HAVE_LINUX_GETAUXVAL
            return hwcap().has_feature(HWCAP_NEON);
#else
            return static_cast<bool>(XSIMD_WITH_NEON);
#endif
        }

        constexpr bool neon64() const noexcept
        {
            return static_cast<bool>(XSIMD_WITH_NEON64);
        }

        inline bool sve() const noexcept
        {
#if XSIMD_TARGET_ARM64 && XSIMD_HAVE_LINUX_GETAUXVAL
            return hwcap().has_feature(HWCAP_SVE);
#else
            return false;
#endif
        }

        inline bool i8mm() const noexcept
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

    private:
#if XSIMD_TARGET_ARM && XSIMD_HAVE_LINUX_GETAUXVAL
        enum class status
        {
            hwcap_valid = 0,
            hwcap2_valid = 1,
        };

        using status_bitset = utils::uint_bitset<status, std::uint32_t>;

        mutable status_bitset m_status {};

        mutable xsimd::linux_auxval m_hwcap {};

        inline xsimd::linux_auxval const& hwcap() const noexcept
        {
            if (!m_status.bit_is_set<status::hwcap_valid>())
            {
                m_hwcap = xsimd::linux_auxval::read(AT_HWCAP);
                m_status.set_bit<status::hwcap_valid>();
            }
            return m_hwcap;
        }

#if XSIMD_TARGET_ARM64
        mutable xsimd::linux_auxval m_hwcap2 {};

        inline xsimd::linux_auxval const& hwcap2() const noexcept
        {
            if (!m_status.bit_is_set<status::hwcap2_valid>())
            {
                m_hwcap2 = xsimd::linux_auxval::read(AT_HWCAP2);
                m_status.set_bit<status::hwcap2_valid>();
            }
            return m_hwcap2;
        }
#endif
#endif // XSIMD_TARGET_ARM && XSIMD_HAVE_LINUX_GETAUXVAL
    };
}
#endif
