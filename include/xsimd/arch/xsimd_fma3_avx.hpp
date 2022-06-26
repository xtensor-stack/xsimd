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

#ifndef XSIMD_FMA3_AVX_HPP
#define XSIMD_FMA3_AVX_HPP

#include "../types/xsimd_fma3_avx_register.hpp"

namespace xsimd
{

    namespace kernel
    {
        using namespace types;

        // fnma
        template <class A>
        inline batch<float, A> XSIMD_CALLCONV fnma(batch<float, A> const& x, batch<float, A> const& y, batch<float, A> const& z, requires_arch<fma3<avx>>) noexcept
        {
            return _mm256_fnmadd_ps(x, y, z);
        }

        template <class A>
        inline batch<double, A> XSIMD_CALLCONV fnma(batch<double, A> const& x, batch<double, A> const& y, batch<double, A> const& z, requires_arch<fma3<avx>>) noexcept
        {
            return _mm256_fnmadd_pd(x, y, z);
        }

        // fnms
        template <class A>
        inline batch<float, A> XSIMD_CALLCONV fnms(batch<float, A> const& x, batch<float, A> const& y, batch<float, A> const& z, requires_arch<fma3<avx>>) noexcept
        {
            return _mm256_fnmsub_ps(x, y, z);
        }

        template <class A>
        inline batch<double, A> XSIMD_CALLCONV fnms(batch<double, A> const& x, batch<double, A> const& y, batch<double, A> const& z, requires_arch<fma3<avx>>) noexcept
        {
            return _mm256_fnmsub_pd(x, y, z);
        }

        // fma
        template <class A>
        inline batch<float, A> XSIMD_CALLCONV fma(batch<float, A> const& x, batch<float, A> const& y, batch<float, A> const& z, requires_arch<fma3<avx>>) noexcept
        {
            return _mm256_fmadd_ps(x, y, z);
        }

        template <class A>
        inline batch<double, A> XSIMD_CALLCONV fma(batch<double, A> const& x, batch<double, A> const& y, batch<double, A> const& z, requires_arch<fma3<avx>>) noexcept
        {
            return _mm256_fmadd_pd(x, y, z);
        }

        // fms
        template <class A>
        inline batch<float, A> XSIMD_CALLCONV fms(batch<float, A> const& x, batch<float, A> const& y, batch<float, A> const& z, requires_arch<fma3<avx>>) noexcept
        {
            return _mm256_fmsub_ps(x, y, z);
        }

        template <class A>
        inline batch<double, A> XSIMD_CALLCONV fms(batch<double, A> const& x, batch<double, A> const& y, batch<double, A> const& z, requires_arch<fma3<avx>>) noexcept
        {
            return _mm256_fmsub_pd(x, y, z);
        }

    }

}

#endif
