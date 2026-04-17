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
#include "./xsimd_cpu_features_arm.hpp"
#include "./xsimd_cpu_features_ppc.hpp"
#include "./xsimd_cpu_features_riscv.hpp"
#include "./xsimd_cpu_features_x86.hpp"
#include "./xsimd_inline.hpp"

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
            ARCH_FIELD_EX(detail::sve<512>, sve512)
            ARCH_FIELD_EX(detail::sve<256>, sve256)
            ARCH_FIELD_EX(detail::sve<128>, sve128)
            ARCH_FIELD_EX(detail::rvv<512>, rvv512)
            ARCH_FIELD_EX(detail::rvv<256>, rvv256)
            ARCH_FIELD_EX(detail::rvv<128>, rvv128)
            ARCH_FIELD(wasm)
            ARCH_FIELD(vsx)

#undef ARCH_FIELD

            XSIMD_INLINE supported_arch() noexcept
            {
#if XSIMD_WITH_WASM
                wasm = 1;
#endif

                // Safe on all platforms, it will be false if non PowerPC.
                const auto ppc_cpu = xsimd::ppc_cpu_features();

                vsx = ppc_cpu.vsx();

                // Safe on all platforms, it will be all false if non risc-v.
                const auto riscv_cpu = xsimd::riscv_cpu_features();

                rvv128 = riscv_cpu.rvv() && (riscv_cpu.rvv_size_bytes() >= (128 / 8));
                rvv256 = riscv_cpu.rvv() && (riscv_cpu.rvv_size_bytes() >= (256 / 8));
                rvv512 = riscv_cpu.rvv() && (riscv_cpu.rvv_size_bytes() >= (512 / 8));

                // Safe on all platforms, it will be all false if non arm.
                const auto arm_cpu = xsimd::arm_cpu_features();

                neon = arm_cpu.neon();
                neon64 = arm_cpu.neon64();
                i8mm_neon64 = arm_cpu.neon64() && arm_cpu.i8mm();
                sve128 = arm_cpu.sve() && (arm_cpu.sve_size_bytes() >= (128 / 8));
                sve256 = arm_cpu.sve() && (arm_cpu.sve_size_bytes() >= (256 / 8));
                sve512 = arm_cpu.sve() && (arm_cpu.sve_size_bytes() >= (512 / 8));

                // Safe on all platforms, it will be all false if non x86.
                const auto x86_cpu = xsimd::x86_cpu_features();

                sse2 = x86_cpu.sse2();
                sse3 = x86_cpu.sse3();
                ssse3 = x86_cpu.ssse3();
                sse4_1 = x86_cpu.sse4_1();
                sse4_2 = x86_cpu.sse4_2();
                fma3_sse42 = x86_cpu.fma3();

                // sse4a not implemented in cpu_id yet
                // xop not implemented in cpu_id yet

                avx = x86_cpu.avx();
                fma3_avx = avx && fma3_sse42;
                fma4 = x86_cpu.fma4();
                avx2 = x86_cpu.avx2();
                avxvnni = x86_cpu.avxvnni();
                fma3_avx2 = avx2 && fma3_sse42;

                avx512f = x86_cpu.avx512f();
                avx512cd = x86_cpu.avx512cd();
                avx512dq = x86_cpu.avx512dq();
                avx512bw = x86_cpu.avx512bw();
                avx512er = x86_cpu.avx512er();
                avx512pf = x86_cpu.avx512pf();
                avx512ifma = x86_cpu.avx512ifma();
                avx512vbmi = x86_cpu.avx512vbmi();
                avx512vbmi2 = x86_cpu.avx512vbmi2();
                avx512vnni_bw = x86_cpu.avx512vnni_bw();
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
