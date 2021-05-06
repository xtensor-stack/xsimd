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

#ifndef XSIMD_ARCH_HPP
#define XSIMD_ARCH_HPP

#include "xsimd_instruction_set.hpp"

namespace xsimd
{

    // forward declaration
    template<class T, size_t N>
    class batch;

    namespace arch
    {
        struct sse
        {
            template<class T>
            using batch = xsimd::batch<T, 128 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 8;
        };

        struct sse2 : sse
        {
            static constexpr size_t alignment = 16;
        };

        struct sse3 : sse2
        {
        };

        // Intel specific
        struct sse4_1 : sse3
        {
        };

        // Intel specific
        struct sse4_2 : sse4_1
        {
        };

        // AMD specific
        struct sse4a : sse3
        {
        };

        struct avx
        {
            template<class T>
            using batch = xsimd::batch<T, 256 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 32;
        };

        struct fma3 : avx
        {
        };

        // AMD specific (very few old processors)
        struct fma4 : avx
        {
        };

        // AMD specific (very few old processors)
        struct xop : fma4
        {
        };

        struct avx2 : avx
        {
        };

        struct avx512
        {
            template<class T>
            using batch = xsimd::batch<T, 512 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 64;
        };

        struct neon64
        {
            template<class T>
            using batch = xsimd::batch<T, 512 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 32;
        };

        struct neon
        {
            template<class T>
            using batch = xsimd::batch<T, 512 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 16;
        };

        struct scalar
        {
            template<class T>
            using batch = xsimd::batch<T, 1>;
            static constexpr size_t alignment = sizeof(void*);
        };

        template <class T, class InstructionSet>
        using batch = typename InstructionSet::template batch<T>;

/***********************
 * X86 instruction set *
 ***********************/

#if XSIMD_X86_INSTR_SET == XSIMD_X86_SSE_VERSION
        using x86 = sse;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_SSE2_VERSION
        using x86 = sse2;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_SSE3_VERSION
        using x86 = sse3;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_SSE4_1_VERSION
        using x86 = sse4_1;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_SSE4_2_VERSION
        using x86 = sse4_2;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_AVX_VERSION
        using x86 = avx;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_FMA3_VERSION
        using x86 = fma3;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_AVX2_VERSION
        using x86 = avx2;
#elif XSIMD_X86_INSTR_SET == XSIMD_X86_AVX512_VERSION
        using x86 = avx512;
#endif

/***************************
 * X86 AMD instruction set *
 ***************************/

#if XSIMD_X86_AMD_INSTR_SET == XSIMD_X86_AMD_SSE4A_VERSION
        using x86 = sse4a;
#elif XSIMD_X86_AMD_INSTR_SET == XSIMD_X86_AMD_FMA4_VERSION
        using x86 = fma4;
#elif XSIMD_X86_AMD_INSTR_SET == XSIMD_X86_AMD_XOP_VERSION
        using x86 = xop;
#endif

/***********************
 * ARM instruction set *
 ***********************/

#if XSIMD_ARM_INSTR_SET == XSIMD_ARM7_NEON_VERSION
        using arm = neon;
#elif XSIMD_ARM_INSTR_SET == XSIMD_ARM8_32_NEON_VERSION
        using arm = neon;
#elif XSIMD_ARM_INSTR_SET == XSIMD_ARM8_64_NEON_VERSION
        using arm = neon64;
#endif

#if XSIMD_X86_INSTR_SET != XSIMD_VERSION_NUMBER_NOT_AVAILABLE
        using default_ = x86;
#elif XSIMD_ARM_INSTR_SET != XSIMD_VERSION_NUMBER_NOT_AVAILABLE
        using default_ = arm;
#else
        using default_ = scalar;
#endif

    }
}

#endif
