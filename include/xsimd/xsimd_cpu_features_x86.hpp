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

#ifndef XSIMD_CPU_FEATURES_X86_HPP
#define XSIMD_CPU_FEATURES_X86_HPP

#include <array>
#include <cstdint>

#include "./config/xsimd_config.hpp"
#include "./utils/bits.hpp"

#if XSIMD_TARGET_X86 && defined(_MSC_VER)
#include <intrin.h> // Contains the definition of __cpuidex
#endif

namespace xsimd
{
    namespace detail
    {
        using cpuid_reg_t = std::array<int, 4>;
        inline cpuid_reg_t get_cpuid(int level, int count = 0) noexcept;

        using xcr0_reg_t = std::uint32_t;
        inline xcr0_reg_t get_xcr0_low() noexcept;
    }

    /*
     * Extended Control Register 0 (XCR0).
     *
     * Operating systems can explicitly disable the usage of instruction set (such
     * as SSE or AVX extensions) by setting an appropriate flag in XCR0 register.
     * This utility parses such bit values.
     *
     * @see https://docs.kernel.org/admin-guide/hw-vuln/gather_data_sampling.html
     */
    class x86_xcr0
    {
    public:
        using reg_t = detail::xcr0_reg_t;

        static constexpr reg_t sse_bit = 1;
        static constexpr reg_t avx_bit = 2;
        static constexpr reg_t avx512_bit = 6;

        /** Parse a XCR0 value into individual components. */
        constexpr explicit x86_xcr0(reg_t low) noexcept
            : m_low(low)
        {
        }

        /**
         * Create a default value with only SSE enabled.
         *
         * AVX and AVX512 strictly require OSXSAVE to be enabled by the OS.
         * If OSXSAVE is disabled (e.g., via bcdedit /set xsavedisable 1), AVX state won't
         * be preserved across context switches, so AVX cannot be used.
         * SSE is therefore the only value safe to assume.
         */
        constexpr static x86_xcr0 safe_default() noexcept
        {
            reg_t low = {};
            low = utils::set_bit<sse_bit>(low);
            return x86_xcr0(low);
        }

        /** Read the XCR0 register from the CPU if on the correct architecture. */
        inline static x86_xcr0 read()
        {
            return x86_xcr0(detail::get_xcr0_low());
        }

        constexpr bool sse_enabled() const noexcept
        {
            return utils::bit_is_set<sse_bit>(m_low);
        }

        constexpr bool avx_enabled() const noexcept
        {
            // Check both SSE and AVX bits even though AVX must imply SSE
            return utils::bit_is_set<sse_bit, avx_bit>(m_low);
        }

        constexpr bool avx512_enabled() const noexcept
        {
            // Check all SSE, AVX, and AVX512 bits even though AVX512 must
            // imply AVX and SSE
            return utils::bit_is_set<sse_bit, avx_bit, avx512_bit>(m_low);
        }

    private:
        reg_t m_low = {};
    };

    /**
     * CPU Identification (CPUID) instruction results.
     *
     * The CPUID instruction provides detailed information about the processor,
     * including supported instruction set extensions (SSE, AVX, AVX-512, etc.).
     * This utility parses CPUID leaf values to detect available CPU features.
     *
     * @see https://en.wikipedia.org/wiki/CPUID
     */
    class x86_cpu_id
    {
    public:
        struct cpu_id_regs
        {
            using reg_t = detail::cpuid_reg_t;

            reg_t reg1 = {};
            reg_t reg7 = {};
            reg_t reg7a = {};
            reg_t reg8 = {};
        };

        /** Parse CpuInfo register values into individual components. */
        constexpr explicit x86_cpu_id(const cpu_id_regs& regs) noexcept
            : m_regs(regs)
        {
        }

        /**
         * Read the CpuId registers from the CPU if on the correct architecture.
         *
         * This is only safe to call if bit 18 of CR4.OSXSAVE has been set.
         *
         * @see cpu_id::osxsave
         */
        inline static x86_cpu_id read()
        {
            cpu_id_regs regs = {};
            // TODO(C++20): Use designated initializer
            regs.reg1 = detail::get_cpuid(0x1);
            regs.reg7 = detail::get_cpuid(0x7);
            regs.reg7a = detail::get_cpuid(0x7, 0x1);
            regs.reg8 = detail::get_cpuid(0x80000001);
            return x86_cpu_id(regs);
        }

        constexpr bool sse2() const noexcept { return utils::bit_is_set<26>(m_regs.reg1[3]); }

        constexpr bool sse3() const noexcept { return utils::bit_is_set<0>(m_regs.reg1[2]); }

        constexpr bool ssse3() const noexcept { return utils::bit_is_set<9>(m_regs.reg1[2]); }

        constexpr bool sse4_1() const noexcept { return utils::bit_is_set<19>(m_regs.reg1[2]); }

        constexpr bool sse4_2() const noexcept { return utils::bit_is_set<20>(m_regs.reg1[2]); }

        constexpr bool fma3() const noexcept { return utils::bit_is_set<12>(m_regs.reg1[2]); }

        /**
         * Indicates whether the OS has enabled extended state management.
         *
         * When true, the OS has set bit 18 (OSXSAVE) in the CR4 control register,
         * enabling the XGETBV/XSETBV instructions to access XCR0 and support
         * processor extended state management using XSAVE/XRSTOR.
         *
         * This value is read from CPUID leaf 0x1, ECX bit 27, which reflects
         * the state of CR4.OSXSAVE.
         */
        constexpr bool osxsave() const noexcept { return utils::bit_is_set<27>(m_regs.reg1[2]); }

        constexpr bool avx() const noexcept { return utils::bit_is_set<28>(m_regs.reg1[2]); }

        constexpr bool avx2() const noexcept { return utils::bit_is_set<5>(m_regs.reg7[1]); }

        constexpr bool avx512f() const noexcept { return utils::bit_is_set<16>(m_regs.reg7[1]); }

        constexpr bool avx512dq() const noexcept { return utils::bit_is_set<17>(m_regs.reg7[1]); }

        constexpr bool avx512ifma() const noexcept { return utils::bit_is_set<21>(m_regs.reg7[1]); }

        constexpr bool avx512pf() const noexcept { return utils::bit_is_set<26>(m_regs.reg7[1]); }

        constexpr bool avx512er() const noexcept { return utils::bit_is_set<27>(m_regs.reg7[1]); }

        constexpr bool avx512cd() const noexcept { return utils::bit_is_set<28>(m_regs.reg7[1]); }

        constexpr bool avx512bw() const noexcept { return utils::bit_is_set<30>(m_regs.reg7[1]); }

        constexpr bool avx512vbmi() const noexcept { return utils::bit_is_set<1>(m_regs.reg7[2]); }

        constexpr bool avx512vbmi2() const noexcept { return utils::bit_is_set<6>(m_regs.reg7[2]); }

        constexpr bool avx512vnni_bw() const noexcept { return utils::bit_is_set<11>(m_regs.reg7[2]); }

        constexpr bool avxvnni() const noexcept { return utils::bit_is_set<4>(m_regs.reg7a[0]); }

        constexpr bool fma4() const noexcept { return utils::bit_is_set<16>(m_regs.reg8[2]); }

    private:
        cpu_id_regs m_regs = {};
    };

    namespace detail
    {
        inline cpuid_reg_t get_cpuid(int level, int count) noexcept
        {
            cpuid_reg_t reg = {};

#if !XSIMD_TARGET_X86
            (void)level;
            (void)count;
            return {}; // All bits to zero

#elif defined(_MSC_VER)
            __cpuidex(reg.data(), level, count);

#elif defined(__INTEL_COMPILER)
            __cpuid(reg.data(), level);

#elif defined(__GNUC__) || defined(__clang__)

#if defined(__i386__) && defined(__PIC__)
            // %ebx may be the PIC register
            __asm__("xchg{l}\t{%%}ebx, %1\n\t"
                    "cpuid\n\t"
                    "xchg{l}\t{%%}ebx, %1\n\t"
                    : "=a"(reg[0]), "=r"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                    : "0"(level), "2"(count));

#else
            __asm__("cpuid\n\t"
                    : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                    : "0"(level), "2"(count));
#endif
#endif
            return reg;
        }

        inline xcr0_reg_t get_xcr0_low() noexcept
        {
#if !XSIMD_TARGET_X86
            return {}; // All bits to zero

#elif defined(_MSC_VER) && _MSC_VER >= 1400
            return static_cast<xcr0_reg_t>(_xgetbv(0));

#elif defined(__GNUC__)
            xcr0_reg_t xcr0 = {};
            __asm__(
                "xorl %%ecx, %%ecx\n"
                "xgetbv\n"
                : "=a"(xcr0)
                :
#if defined(__i386__)
                : "ecx", "edx"
#else
                : "rcx", "rdx"
#endif
            );
            return xcr0;

#else /* _MSC_VER < 1400 */
#error "_MSC_VER < 1400 is not supported"
#endif /* _MSC_VER && _MSC_VER >= 1400 */
        };
    }
}
#endif
