#ifndef XSIMD_NEON64_REGISTER_HPP
#define XSIMD_NEON64_REGISTER_HPP

#include "xsimd_neon_register.hpp"

namespace xsimd
{
    struct neon64 : neon
    {
        static constexpr bool supported() { return XSIMD_WITH_NEON64; }
        static constexpr bool available() { return true; }
        static constexpr bool requires_alignment() { return true; }
        static constexpr std::size_t alignment() { return 16; }
        static constexpr unsigned version() { return generic::version(8, 1, 0); }
        static constexpr char const* name() { return "arm64+neon"; }
    };

#if XSIMD_WITH_NEON64

    namespace types
    {
        XSIMD_DECLARE_SIMD_REGISTER_ALIAS(neon64, neon);
        XSIMD_DECLARE_SIMD_REGISTER(double, neon64, float64x2_t);

        template <class T>
        struct get_bool_simd_register<T, neon64>
            : detail::neon_bool_simd_register<T, neon64>
        {
        };
    }

#endif

}

#endif


