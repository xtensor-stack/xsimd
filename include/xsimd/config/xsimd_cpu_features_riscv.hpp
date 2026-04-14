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

#if XSIMD_TARGET_RISCV && XSIMD_HAVE_LINUX_GETAUXVAL
#include "../utils/bits.hpp"
#include "./xsimd_getauxval.hpp"

// HWCAP_XXX masks to use on getauxval results.
// Header does not exists on all architectures and masks are architecture
// specific.
#include <asm/hwcap.h>
#endif // XSIMD_TARGET_RISCV && XSIMD_HAVE_LINUX_GETAUXVAL

namespace xsimd
{
    class riscv_cpu_features
    {
    public:
        riscv_cpu_features() noexcept = default;

        inline bool rvv() const noexcept
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

    private:
#if XSIMD_TARGET_RISCV && XSIMD_HAVE_LINUX_GETAUXVAL
        enum class status
        {
            hwcap_valid = 0,
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
#endif
    };
}

#endif
