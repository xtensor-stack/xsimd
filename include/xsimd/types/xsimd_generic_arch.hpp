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

#ifndef XSIMD_GENERIC_ARCH_HPP
#define XSIMD_GENERIC_ARCH_HPP

/**
 * @defgroup arch
 * */
namespace xsimd
{
    struct generic
    {
        static constexpr bool supported() { return true; }
        static constexpr bool available() { return true; }
        static constexpr bool requires_alignment() { return false; }

    protected:

        static constexpr unsigned version(unsigned major, unsigned minor, unsigned patch) { return major * 10000u + minor * 100u + patch; }
    };
}

#endif

