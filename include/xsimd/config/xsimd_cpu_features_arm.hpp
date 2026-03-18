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

#if XSIMD_WITH_LINUX_GETAUXVAL
#include "./xsimd_getauxval.hpp"
#endif

namespace xsimd
{
    /**
     * An opinionated CPU feature detection utility for ARM.
     *
     * Combines compile-time knowledge with runtime detection when available.
     * On Linux, runtime detection uses getauxval to query the auxiliary vector.
     * On other platforms, only compile-time information is used.
     *
     * This is well defined on all architectures. It will always return false on
     * non-ARM architectures.
     */
    class arm_cpu_features
    {
    public:
        arm_cpu_features() noexcept = default;

        inline bool neon() const noexcept
        {
#if XSIMD_TARGET_ARM && !XSIMD_TARGET_ARM64 && XSIMD_WITH_LINUX_GETAUXVAL
            return get_hwcap().all_bits_set<linux_hwcap_traits::aux::neon>();
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
#if XSIMD_TARGET_ARM64 && XSIMD_WITH_LINUX_GETAUXVAL
            return get_hwcap().all_bits_set<linux_hwcap_traits::aux::sve>();
#else
            return false;
#endif
        }

        inline bool i8mm() const noexcept
        {
#if XSIMD_TARGET_ARM64 && XSIMD_WITH_LINUX_GETAUXVAL
            return get_hwcap2().all_bits_set<linux_hwcap2_traits::aux::i8mm>();
#else
            return false;
#endif
        }

    private:
#if XSIMD_TARGET_ARM && XSIMD_WITH_LINUX_GETAUXVAL
        enum class status
        {
            hwcap_valid = 0,
            hwcap2_valid = 1,
        };

        using status_bitset = utils::uint_bitset<status, std::uint32_t>;

        mutable status_bitset m_status {};

        mutable xsimd::linux_hwcap m_hwcap {};

        inline xsimd::linux_hwcap const& get_hwcap() const noexcept
        {
            if (!m_status.bit_is_set<status::hwcap_valid>())
            {
                m_hwcap = xsimd::linux_hwcap::read();
                m_status.set_bit<status::hwcap_valid>();
            }
            return m_hwcap;
        }

#if XSIMD_TARGET_ARM64
        mutable xsimd::linux_hwcap2 m_hwcap2 {};
        inline xsimd::linux_hwcap2 const& get_hwcap2() const noexcept
        {
            if (!m_status.bit_is_set<status::hwcap2_valid>())
            {
                m_hwcap2 = xsimd::linux_hwcap2::read();
                m_status.set_bit<status::hwcap2_valid>();
            }
            return m_hwcap2;
        }
#endif
#endif // XSIMD_TARGET_ARM && XSIMD_WITH_LINUX_GETAUXVAL
    };
}
#endif
