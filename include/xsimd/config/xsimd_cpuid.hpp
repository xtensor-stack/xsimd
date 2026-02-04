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

#ifndef XSIMD_CPUID_HPP
#define XSIMD_CPUID_HPP

#include "../types/xsimd_all_registers.hpp"
#include "./xsimd_cpu_features_x86.hpp"
#include "xsimd_inline.hpp"

#if defined(__linux__) && (defined(__ARM_NEON) || defined(_M_ARM) || defined(__riscv_vector))
#include <asm/hwcap.h>
#include <sys/auxv.h>

#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif

#endif

namespace xsimd
{
    namespace detail
    {
        struct supported_arch
        {

#define ARCH_FIELD_EX(arch, field_name) \
    unsigned field_name = 0;            \
    XSIMD_INLINE bool has(::xsimd::arch) const { return this->field_name; }

#define ARCH_FIELD_EX_REUSE(arch, field_name) \
    XSIMD_INLINE bool has(::xsimd::arch) const { return this->field_name; }

#define ARCH_FIELD(name) ARCH_FIELD_EX(name, name)

            ARCH_FIELD(sse2)
            ARCH_FIELD(sse3)

            ARCH_FIELD(ssse3)
            ARCH_FIELD(sse4_1)
            ARCH_FIELD(sse4_2)
            // ARCH_FIELD(sse4a)
            ARCH_FIELD_EX(fma3<::xsimd::sse4_2>, fma3_sse42)
            ARCH_FIELD(fma4)
            // ARCH_FIELD(xop)
            ARCH_FIELD(avx)
            ARCH_FIELD_EX(fma3<::xsimd::avx>, fma3_avx)
            ARCH_FIELD(avx2)
            ARCH_FIELD(avxvnni)
            ARCH_FIELD_EX(fma3<::xsimd::avx2>, fma3_avx2)
            ARCH_FIELD(avx512f)
            ARCH_FIELD(avx512cd)
            ARCH_FIELD(avx512dq)
            ARCH_FIELD(avx512bw)
            ARCH_FIELD(avx512er)
            ARCH_FIELD(avx512pf)
            ARCH_FIELD(avx512ifma)
            ARCH_FIELD(avx512vbmi)
            ARCH_FIELD(avx512vbmi2)
            ARCH_FIELD_EX(avx512vnni<::xsimd::avx512bw>, avx512vnni_bw)
            ARCH_FIELD_EX(avx512vnni<::xsimd::avx512vbmi2>, avx512vnni_vbmi2)
            ARCH_FIELD(neon)
            ARCH_FIELD(neon64)
            ARCH_FIELD_EX(i8mm<::xsimd::neon64>, i8mm_neon64)
            ARCH_FIELD_EX(detail::sve<512>, sve)
            ARCH_FIELD_EX_REUSE(detail::sve<256>, sve)
            ARCH_FIELD_EX_REUSE(detail::sve<128>, sve)
            ARCH_FIELD_EX(detail::rvv<512>, rvv)
            ARCH_FIELD_EX_REUSE(detail::rvv<256>, rvv)
            ARCH_FIELD_EX_REUSE(detail::rvv<128>, rvv)
            ARCH_FIELD(wasm)
            ARCH_FIELD(vsx)

#undef ARCH_FIELD

            XSIMD_INLINE supported_arch() noexcept
            {
#if XSIMD_WITH_WASM
                wasm = 1;
#endif

#if XSIMD_WITH_VSX
                vsx = 1;
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
                neon = 1;
                neon64 = 1;
#if defined(__linux__) && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 18)
                i8mm_neon64 = bool(getauxval(AT_HWCAP2) & HWCAP2_I8MM);
                sve = bool(getauxval(AT_HWCAP) & HWCAP_SVE);
#endif

#elif defined(__ARM_NEON) || defined(_M_ARM)

#if defined(__linux__) && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 18)
                neon = bool(getauxval(AT_HWCAP) & HWCAP_NEON);
#endif

#elif defined(__riscv_vector) && defined(__riscv_v_fixed_vlen) && __riscv_v_fixed_vlen > 0

#if defined(__linux__) && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 18)
#ifndef HWCAP_V
#define HWCAP_V (1 << ('V' - 'A'))
#endif
                rvv = bool(getauxval(AT_HWCAP) & HWCAP_V);
#endif
#endif
                // Safe on all platforms, we simply be false
                const auto cpuid = xsimd::x86_cpu_id::read();
                const auto xcr0 = cpuid.osxsave() ? x86_xcr0::read() : x86_xcr0::safe_default();

                sse2 = cpuid.sse2() && xcr0.sse_enabled();
                sse3 = cpuid.sse3() && xcr0.sse_enabled();
                ssse3 = cpuid.ssse3() && xcr0.sse_enabled();
                sse4_1 = cpuid.sse4_1() && xcr0.sse_enabled();
                sse4_2 = cpuid.sse4_2() && xcr0.sse_enabled();
                fma3_sse42 = cpuid.fma3() && xcr0.sse_enabled();

                // sse4a not implemented in cpu_id yet
                // xop not implemented in cpu_id yet

                avx = cpuid.avx() && xcr0.avx_enabled();
                fma3_avx = avx && fma3_sse42;
                fma4 = cpuid.fma4() && xcr0.avx_enabled();
                avx2 = cpuid.avx2() && xcr0.avx_enabled();
                avxvnni = cpuid.avxvnni() && xcr0.avx_enabled();
                fma3_avx2 = avx2 && fma3_sse42;

                avx512f = cpuid.avx512f() && xcr0.avx512_enabled();
                avx512cd = cpuid.avx512cd() && xcr0.avx512_enabled();
                avx512dq = cpuid.avx512dq() && xcr0.avx512_enabled();
                avx512bw = cpuid.avx512bw() && xcr0.avx512_enabled();
                avx512er = cpuid.avx512er() && xcr0.avx512_enabled();
                avx512pf = cpuid.avx512pf() && xcr0.avx512_enabled();
                avx512ifma = cpuid.avx512ifma() && xcr0.avx512_enabled();
                avx512vbmi = cpuid.avx512vbmi() && xcr0.avx512_enabled();
                avx512vbmi2 = cpuid.avx512vbmi2() && xcr0.avx512_enabled();
                avx512vnni_bw = cpuid.avx512vnni_bw() && xcr0.avx512_enabled();
                avx512vnni_vbmi2 = avx512vbmi2 && avx512vnni_bw;
            }
        };
    } // namespace detail

    XSIMD_INLINE detail::supported_arch available_architectures() noexcept
    {
        static detail::supported_arch supported;
        return supported;
    }
}

#endif
