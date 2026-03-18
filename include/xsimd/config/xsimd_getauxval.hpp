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

#ifndef XSIMD_GETAUXVAL_HPP
#define XSIMD_GETAUXVAL_HPP

#include "../utils/bits.hpp"
#include "./xsimd_config.hpp"

#if XSIMD_WITH_LINUX_GETAUXVAL
#include <sys/auxv.h> // getauxval
#endif

namespace xsimd
{
    namespace detail
    {
        using linux_getauxval_t = unsigned long;

        inline linux_getauxval_t linux_getauxval(linux_getauxval_t type) noexcept;

        /**
         * Base class for getauxval querying.
         */
        template <linux_getauxval_t type, typename A>
        class linux_auxval : private utils::uint_bitset<A, linux_getauxval_t>
        {
            using bitset_t = utils::uint_bitset<A, linux_getauxval_t>;

        public:
            using aux = A;

            inline static linux_auxval read()
            {
                return bitset_t(linux_getauxval(type));
            }

            /** Create a value which returns false to everything. */
            constexpr linux_auxval() noexcept = default;

            using bitset_t::all_bits_set;
        };

        template <typename Traits>
        using make_auxiliary_val_t = linux_auxval<Traits::type, typename Traits::aux>;
    }

    /*
     * Hardware Capabilities Register (HWCAP) for Linux.
     *
     * On Linux systems, the kernel exposes some CPU features through the
     * auxiliary vector, which can be queried via `getauxval(AT_HWCAP)`.
     * This utility parses such bit values.
     *
     * @see https://www.kernel.org/doc/Documentation/arm64/elf_hwcaps.txt
     */
    struct linux_hwcap_traits
    {
#if XSIMD_WITH_LINUX_GETAUXVAL
        static constexpr detail::linux_getauxval_t type = AT_HWCAP;
#else
        static constexpr detail::linux_getauxval_t type = 0;
#endif

        enum class aux
        {
#if XSIMD_WITH_LINUX_GETAUXVAL
#if XSIMD_TARGET_ARM64
            /** Scalable Vector Extension. */
            sve = 22,
#elif XSIMD_TARGET_ARM
            /** Neon vector extension. */
            neon = 12,
#endif
#endif
        };
    };

    using linux_hwcap = detail::make_auxiliary_val_t<linux_hwcap_traits>;

    /*
     * Extended Hardware Capabilities Register (HWCAP2) for Linux.
     *
     * On Linux systems, the kernel exposes some CPU additional features through the
     * auxiliary vector, which can be queried via `getauxval(AT_HWCAP2)`.
     *
     * @see https://www.kernel.org/doc/Documentation/arm64/elf_hwcaps.txt
     */
    struct linux_hwcap2_traits
    {
#if XSIMD_WITH_LINUX_GETAUXVAL
        static constexpr detail::linux_getauxval_t type = AT_HWCAP2;
#else
        static constexpr detail::linux_getauxval_t type = 0;
#endif

        enum class aux
        {
#if XSIMD_WITH_LINUX_GETAUXVAL
#if XSIMD_TARGET_ARM64
            /** 8 bits integer matrix multiplication. */
            i8mm = 13,
#endif
#endif
        };
    };

    using linux_hwcap2 = detail::make_auxiliary_val_t<linux_hwcap2_traits>;

    /********************
     *  Implementation  *
     ********************/

    namespace detail
    {
#if XSIMD_WITH_LINUX_GETAUXVAL
        inline linux_getauxval_t linux_getauxval(linux_getauxval_t type) noexcept
        {
            return getauxval(type);
        }
#else
        inline linux_getauxval_t linux_getauxval(linux_getauxval_t type) noexcept
        {
            return {}; // All bits set to 0
        }
#endif
    }
}

#endif
