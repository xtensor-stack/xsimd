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

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>

#include <doctest/doctest.h>

#include "xsimd/xsimd.hpp"

#define CHECK_IMPLICATION(a, b) CHECK_UNARY(!(a) || (b))

namespace detail
{
    void check_env_flag(const char* env_var, const char* feature_name, bool actual)
    {
        if (const char* val = std::getenv(env_var))
        {
            // Doctest struggles with string literals and const char *
            // TODO(c++20): use std::format
            auto msg = std::string(env_var) + " = " + val + ", " + feature_name + " = " + (actual ? "true" : "false");
            INFO(msg);
            CHECK_EQ(actual, val[0] == '1');
        }
    }

    // TODO(c++23): use str.contains
    bool contains(const std::string& haystack, const char* needle)
    {
        return haystack.find(needle) != std::string::npos;
    }
}

#define CHECK_ENV_FEATURE(env_var, feature) detail::check_env_flag(env_var, #feature, feature)

/**
 * Tests that x86_cpu_features respects the architectural implication chains.
 *
 * These are "always true" assertions: if a higher feature is reported, all
 * features it architecturally implies must also be reported. The test reads
 * the current CPU's features at runtime and verifies every implication.
 */
TEST_CASE("[cpu_features] x86 implication chains")
{
    xsimd::x86_cpu_features cpu;

    // SSE implication chain
    CHECK_IMPLICATION(cpu.sse4_2(), cpu.sse4_1());
    CHECK_IMPLICATION(cpu.sse4_1(), cpu.ssse3());
    CHECK_IMPLICATION(cpu.ssse3(), cpu.sse3());
    CHECK_IMPLICATION(cpu.sse3(), cpu.sse2());

    // AVX implication chain
    CHECK_IMPLICATION(cpu.avx(), cpu.sse4_2());
    CHECK_IMPLICATION(cpu.avx2(), cpu.avx());
    CHECK_IMPLICATION(cpu.fma4(), cpu.avx());
    CHECK_IMPLICATION(cpu.fma3(), cpu.avx());

    // AVX-512 iplication chain
    CHECK_IMPLICATION(cpu.avx512f(), cpu.avx2());
    CHECK_IMPLICATION(cpu.avx512dq(), cpu.avx512f());
    CHECK_IMPLICATION(cpu.avx512ifma(), cpu.avx512f());
    CHECK_IMPLICATION(cpu.avx512pf(), cpu.avx512f());
    CHECK_IMPLICATION(cpu.avx512er(), cpu.avx512f());
    CHECK_IMPLICATION(cpu.avx512cd(), cpu.avx512f());
    CHECK_IMPLICATION(cpu.avx512bw(), cpu.avx512f());
    CHECK_IMPLICATION(cpu.avx512vbmi(), cpu.avx512f());
    CHECK_IMPLICATION(cpu.avx512vbmi2(), cpu.avx512f());
    CHECK_IMPLICATION(cpu.avx512vnni_bw(), cpu.avx512bw());
    CHECK_IMPLICATION(cpu.avxvnni(), cpu.avx2());
}

TEST_CASE("[cpu_features] x86 manufacturer from environment")
{
    xsimd::x86_cpu_features cpu;

    const char* val = std::getenv("XSIMD_TEST_CPU_ASSUME_MANUFACTURER");
    if (val)
    {
        struct entry
        {
            const char* name;
            xsimd::x86_manufacturer value;
        };
        std::array<entry, 9> manufacturers = { {
            { "intel", xsimd::x86_manufacturer::intel },
            { "amd", xsimd::x86_manufacturer::amd },
            { "via", xsimd::x86_manufacturer::via },
            { "zhaoxin", xsimd::x86_manufacturer::zhaoxin },
            { "hygon", xsimd::x86_manufacturer::hygon },
            { "transmeta", xsimd::x86_manufacturer::transmeta },
            { "elbrus", xsimd::x86_manufacturer::elbrus },
            { "microsoft_vpc", xsimd::x86_manufacturer::microsoft_vpc },
            { "unknown", xsimd::x86_manufacturer::unknown },
        } };

        auto manufacturer = cpu.known_manufacturer();
        const std::string allowed(val);
        bool match = std::any_of(manufacturers.begin(), manufacturers.end(), [&](const entry& e)
                                 { return e.value == manufacturer && detail::contains(allowed, e.name); });

        auto const msg = std::string("XSIMD_TEST_CPU_ASSUME_MANUFACTURER = ") + val
            + ", actual = " + xsimd::x86_manufacturer_name(manufacturer);
        INFO(msg);
        CHECK_UNARY(match);
    }
}

TEST_CASE("[cpu_features] x86 features from environment")
{
    xsimd::x86_cpu_features cpu;

    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_SSE2", cpu.sse2());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_SSE3", cpu.sse3());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_SSSE3", cpu.ssse3());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_SSE4_1", cpu.sse4_1());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_SSE4_2", cpu.sse4_2());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_FMA3", cpu.fma3());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_FMA4", cpu.fma4());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_AVX", cpu.avx());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_AVX2", cpu.avx2());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_AVX512F", cpu.avx512f());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_AVX512BW", cpu.avx512bw());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_AVX512CD", cpu.avx512cd());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_AVX512DQ", cpu.avx512dq());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_AVXVNNI", cpu.avxvnni());
}

TEST_CASE("[cpu_features] arm implication chains")
{
    xsimd::arm_cpu_features cpu;

    CHECK_IMPLICATION(cpu.neon64(), cpu.neon());
    CHECK_IMPLICATION(cpu.sve(), cpu.neon64());
    CHECK_IMPLICATION(cpu.sve(), cpu.sve_size_bytes() >= (128 / 8));
    CHECK_IMPLICATION(cpu.i8mm(), cpu.neon64());
}

TEST_CASE("[cpu_features] arm features from environment")
{
    xsimd::arm_cpu_features cpu;

    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_NEON", cpu.neon());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_NEON64", cpu.neon64());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_SVE", cpu.sve());
    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_I8MM", cpu.i8mm());
}

TEST_CASE("[cpu_features] risc-v implication chains")
{
    xsimd::riscv_cpu_features cpu;

    CHECK_IMPLICATION(cpu.rvv(), cpu.rvv_size_bytes() >= (128 / 8));
}

TEST_CASE("[cpu_features] risc-v features from environment")
{
    xsimd::riscv_cpu_features cpu;

    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_RVV", cpu.rvv());
}

TEST_CASE("[cpu_features] ppc features from environment")
{
    xsimd::ppc_cpu_features cpu;

    CHECK_ENV_FEATURE("XSIMD_TEST_CPU_ASSUME_VSX", cpu.vsx());
}
