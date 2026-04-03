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

#include "./xsimd_config.hpp"

#if XSIMD_HAVE_LINUX_GETAUXVAL
#include <sys/auxv.h> // getauxval
#endif

namespace xsimd
{
    namespace detail
    {
        using linux_getauxval_t = unsigned long;

        inline linux_getauxval_t linux_getauxval(linux_getauxval_t type) noexcept;
    }

    /*
     * Holds the value of a Linux auxiliary vector entry (e.g. AT_HWCAP).
     *
     * On Linux systems, the kernel exposes some CPU features through the
     * auxiliary vector, which can be queried via `getauxval(AT_HWCAP)`.
     * Well defined on all platforms, and will return always falsw on
     * non-linux platforms.
     *
     * Usage:
     *   auto hwcap = linux_auxval::read(AT_HWCAP);
     *   bool neon = hwcap.has_feature(HWCAP_NEON);
     *
     * @see https://www.kernel.org/doc/Documentation/arm64/elf_hwcaps.txt
     */
    class linux_auxval
    {
    private:
        using getauxval_t = detail::linux_getauxval_t;

    public:
        constexpr linux_auxval() noexcept = default;

        inline static linux_auxval read(getauxval_t type) noexcept
        {
            return linux_auxval(detail::linux_getauxval(type));
        }

        constexpr bool has_feature(getauxval_t feat) const noexcept
        {
            return (m_auxval & feat) == feat;
        }

    private:
        getauxval_t m_auxval = {};

        constexpr explicit linux_auxval(getauxval_t v) noexcept
            : m_auxval(v)
        {
        }
    };

    /********************
     *  Implementation  *
     ********************/

    namespace detail
    {
#if XSIMD_HAVE_LINUX_GETAUXVAL
        inline linux_getauxval_t linux_getauxval(linux_getauxval_t type) noexcept
        {
            return getauxval(type);
        }
#else
        inline linux_getauxval_t linux_getauxval(linux_getauxval_t) noexcept
        {
            return {}; // All bits set to 0
        }
#endif
    }
}

#endif
