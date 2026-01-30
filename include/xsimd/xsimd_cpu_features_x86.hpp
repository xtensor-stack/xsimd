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

#include <cstdint>

#include "./config/xsimd_config.hpp"

#if XSIMD_TARGET_X86 && defined(_MSC_VER)
// Contains the definition of __cpuidex
#include <intrin.h>
#endif

namespace xsimd
{
    namespace detail
    {
        template <typename I>
        constexpr I make_bit_mask(I bit)
        {
            return static_cast<I>(I { 1 } << bit);
        }

        template <typename I, typename... Args>
        constexpr I make_bit_mask(I bit, Args... bits)
        {
            return make_bit_mask<I>(bit) | make_bit_mask<I>(static_cast<I>(bits)...);
        }

        template <int... Bits, typename I>
        constexpr bool bit_is_set(I value)
        {
            constexpr I mask = make_bit_mask<I>(static_cast<I>(Bits)...);
            return (value & mask) == mask;
        }

        inline void get_cpuid(int reg[4], int level, int count = 0) noexcept;

        inline std::uint32_t get_xcr0_low() noexcept;
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
        using reg_t = std::uint32_t;

        /** Parse a XCR0 value into individual components. */
        constexpr explicit x86_xcr0(reg_t low) noexcept
            : m_low(low)
        {
        }

        /** Create an object that has all features set to false. */
        static constexpr x86_xcr0 make_false()
        {
            return x86_xcr0(0);
        }

        /** Read the XCR0 register from the CPU if on the correct architecture. */
        inline static x86_xcr0 read()
        {
            return x86_xcr0(detail::get_xcr0_low());
        }

        constexpr bool sse_state_os_enabled() const noexcept
        {
            return detail::bit_is_set<1>(m_low);
        }

        constexpr bool avx_state_os_enabled() const noexcept
        {
            // Check both SSE and AVX bits even though AVX must imply SSE
            return detail::bit_is_set<1, 2>(m_low);
        }

        constexpr bool avx512_state_os_enabled() const noexcept
        {
            // Check all SSE, AVX, and AVX52 bits even though AVX512 must
            // imply AVX and SSE
            return detail::bit_is_set<1, 2, 6>(m_low);
        }

    private:
        std::uint32_t m_low = {};
    };

    class x86_cpu_id
    {
    public:
        struct cpu_id_regs
        {
            using reg_t = int[4];

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
            detail::get_cpuid(regs.reg1, 0x1);
            detail::get_cpuid(regs.reg7, 0x7);
            detail::get_cpuid(regs.reg7a, 0x7, 0x1);
            detail::get_cpuid(regs.reg8, 0x80000001);
            return x86_cpu_id(regs);
        }

        constexpr bool sse2() const noexcept
        {
            return detail::bit_is_set<26>(m_regs.reg1[3]);
        }

        constexpr bool sse3() const noexcept
        {
            return detail::bit_is_set<0>(m_regs.reg1[2]);
        }

        constexpr bool ssse3() const noexcept
        {
            return detail::bit_is_set<9>(m_regs.reg1[2]);
        }

        constexpr bool sse4_1() const noexcept
        {
            return detail::bit_is_set<19>(m_regs.reg1[2]);
        }

        constexpr bool sse4_2() const noexcept
        {
            return detail::bit_is_set<20>(m_regs.reg1[2]);
        }

        constexpr bool fma3() const noexcept
        {
            return detail::bit_is_set<12>(m_regs.reg1[2]);
        }

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
        constexpr bool osxsave() const noexcept
        {
            return detail::bit_is_set<27>(m_regs.reg1[2]);
        }

        constexpr bool avx() const noexcept
        {
            return detail::bit_is_set<28>(m_regs.reg1[2]);
        }

        constexpr bool avx2() const noexcept
        {
            return detail::bit_is_set<5>(m_regs.reg7[1]);
        }

        constexpr bool avx512f() const noexcept
        {
            return detail::bit_is_set<16>(m_regs.reg7[1]);
        }

        constexpr bool avx512dq() const noexcept
        {
            return detail::bit_is_set<17>(m_regs.reg7[1]);
        }

        constexpr bool avx512ifma() const noexcept
        {
            return detail::bit_is_set<21>(m_regs.reg7[1]);
        }

        constexpr bool avx512pf() const noexcept
        {
            return detail::bit_is_set<26>(m_regs.reg7[1]);
        }

        constexpr bool avx512er() const noexcept
        {
            return detail::bit_is_set<27>(m_regs.reg7[1]);
        }

        constexpr bool avx512cd() const noexcept
        {
            return detail::bit_is_set<28>(m_regs.reg7[1]);
        }

        constexpr bool avx512bw() const noexcept
        {
            return detail::bit_is_set<30>(m_regs.reg7[1]);
        }

        constexpr bool avx512vbmi() const noexcept
        {
            return detail::bit_is_set<1>(m_regs.reg7[2]);
        }

        constexpr bool avx512vbmi2() const noexcept
        {
            return detail::bit_is_set<6>(m_regs.reg7[2]);
        }

        constexpr bool avx512vnni_bw() const noexcept
        {
            return detail::bit_is_set<11>(m_regs.reg7[2]);
        }

        constexpr bool avxvnni() const noexcept
        {
            return detail::bit_is_set<4>(m_regs.reg7a[0]);
        }

        constexpr bool fma4() const noexcept
        {
            return detail::bit_is_set<16>(m_regs.reg8[2]);
        }

    private:
        cpu_id_regs m_regs = {};
    };

    namespace detail
    {
        inline void get_cpuid(int reg[4], int level, int count) noexcept
        {
#if !XSIMD_TARGET_X86
            reg[0] = 0;
            reg[1] = 0;
            reg[2] = 0;
            reg[3] = 0;
            (void)level;
            (void)count;

#elif defined(_MSC_VER)
            __cpuidex(reg, level, count);

#elif defined(__INTEL_COMPILER)
            __cpuid(reg, level);

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
        }

        inline std::uint32_t get_xcr0_low() noexcept
        {
#if !XSIMD_TARGET_X86
            return {}; // return 0;

#elif defined(_MSC_VER) && _MSC_VER >= 1400
            return static_cast<std::uint32_t>(_xgetbv(0));

#elif defined(__GNUC__)
            std::uint32_t xcr0 = {};
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
