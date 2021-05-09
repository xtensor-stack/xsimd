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
            static const unsigned version = XSIMD_X86_SSE_VERSION;
            static constexpr bool configured = XSIMD_X86_INSTR_SET >= version;

            template<class T>
            using batch = xsimd::batch<T, 128 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 8;
        };

        struct sse2 : sse
        {
            static const unsigned version = XSIMD_X86_SSE2_VERSION;
            static constexpr bool configured = XSIMD_X86_INSTR_SET >= version;

            static constexpr size_t alignment = 16;
        };

        struct sse3 : sse2
        {
            static const unsigned version = XSIMD_X86_SSE3_VERSION;
            static constexpr bool configured = XSIMD_X86_INSTR_SET >= version;
        };

        // Intel specific
        struct sse4_1 : sse3
        {
            static const unsigned version = XSIMD_X86_SSE4_1_VERSION;
            static constexpr bool configured = XSIMD_X86_INSTR_SET >= version;
        };

        // Intel specific
        struct sse4_2 : sse4_1
        {
            static const unsigned version = XSIMD_X86_SSE4_2_VERSION;
            static constexpr bool configured = XSIMD_X86_INSTR_SET >= version;
        };

        // AMD specific
        struct sse4a : sse3
        {
            static const unsigned version = XSIMD_X86_AMD_SSE4A_VERSION;
            static constexpr bool configured = XSIMD_X86_AMD_INSTR_SET >= version;
        };

        struct avx
        {
            static const unsigned version = XSIMD_X86_AVX_VERSION;
            static constexpr bool configured = XSIMD_X86_INSTR_SET >= version;

            template<class T>
            using batch = xsimd::batch<T, 256 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 32;
        };

        struct fma3 : avx
        {
            static const unsigned version = XSIMD_X86_FMA3_VERSION;
            static constexpr bool configured = XSIMD_X86_INSTR_SET >= version;
        };

        // AMD specific (very few old processors)
        struct fma4 : avx
        {
            static const unsigned version = XSIMD_X86_AMD_FMA4_VERSION;
            static constexpr bool configured = XSIMD_X86_AMD_INSTR_SET >= version;
        };

        // AMD specific (very few old processors)
        struct xop : fma4
        {
            static const unsigned version = XSIMD_X86_AMD_XOP_VERSION;
            static constexpr bool configured = XSIMD_X86_AMD_INSTR_SET >= version;
        };

        struct avx2 : avx
        {
            static const unsigned version = XSIMD_X86_AVX2_VERSION;
            static constexpr bool configured = XSIMD_X86_INSTR_SET >= version;
        };

        struct avx512
        {
            static const unsigned version = XSIMD_X86_AVX512_VERSION;
            static constexpr bool configured = XSIMD_X86_INSTR_SET >= version;

            template<class T>
            using batch = xsimd::batch<T, 512 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 64;
        };

        struct neon
        {
            static const unsigned version = XSIMD_ARM7_NEON_VERSION;
            static constexpr bool configured = XSIMD_ARM_INSTR_SET >= version;

            template<class T>
            using batch = xsimd::batch<T, 512 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 16;
        };

        struct neon64
        {
            static const unsigned version = XSIMD_ARM8_64_NEON_VERSION;
            static constexpr bool configured = XSIMD_ARM_INSTR_SET >= version;

            template<class T>
            using batch = xsimd::batch<T, 512 / ( 8 * sizeof(T))>;
            static constexpr size_t alignment = 32;
        };

        struct scalar
        {
            static const unsigned version = XSIMD_VERSION_NUMBER_AVAILABLE;
            static constexpr bool configured = true;

            template<class T>
            using batch = xsimd::batch<T, 1>;
            static constexpr size_t alignment = sizeof(void*);
        };

        struct unavailable
        {
            static const unsigned version = XSIMD_VERSION_NUMBER_NOT_AVAILABLE;
            static constexpr bool configured = false;
        };

        namespace detail
        {
            // Checks whether T appears in Tys.
            template<class T, class... Tys>
            struct contains;

            template<class T>
            struct contains<T> : std::false_type
            {
            };

            template<class T, class Ty, class... Tys>
            struct contains<T, Ty, Tys...> : std::conditional<std::is_same<Ty, T>::value, std::true_type, contains<T, Tys...>>::type
            {
            };

            template<class... Archs>
            struct is_sorted;

            template<>
            struct is_sorted<> : std::true_type
            {
            };

            template<class Arch>
            struct is_sorted<Arch> : std::true_type
            {
            };

            template<class A0, class A1, class... Archs>
            struct is_sorted<A0, A1, Archs...> : std::conditional<(A0::version >= A1::version), is_sorted<Archs...>, std::false_type>::type
            {
            };

        }


        // An arch_list is a list of architectures, sorted by version number.
        template<class... Archs>
        struct arch_list {
#ifndef NDEBUG
          static_assert(detail::is_sorted<Archs...>::value, "architecture list must be sorted by version");
#endif

          template<class Arch>
          using add = arch_list<Archs..., Arch>;

          template<class... OtherArchs>
          using extend = arch_list<Archs..., OtherArchs...>;

          template<class Arch>
          static constexpr bool contains() {
            return detail::contains<Arch, Archs...>::value;
          }
        };


        namespace detail
        {
            // Pick the best architecture in arch_list L, which is the last
            // because architectures are sorted by version.
            template<class L>
            struct best;

            template<>
            struct best<arch_list<>>
            {
                using type = unavailable;
            };

            template<class Arch, class... Archs>
            struct best<arch_list<Arch, Archs...>>
            {
                using type = Arch;
            };

            // Filter archlists Archs, picking only configured archs and adding
            // them to L.
            template<class L, class... Archs>
            struct configured_helper;

            template<class L>
            struct configured_helper<L, arch_list<>>
            {
                using type = L;
            };

            template<class L, class Arch, class... Archs>
            struct configured_helper<L, arch_list<Arch, Archs...>> : configured_helper<typename std::conditional<Arch::configured, typename L::template add<Arch>, L >::type, arch_list<Archs...>>
            {
            };

            template<class... Archs>
            struct configured : configured_helper<arch_list<>, Archs...>
            {
            };

            // Joins all arch_list Archs in a single arch_list.
            template<class... Archs>
            struct join;

            template<class Arch>
            struct join<Arch>
            {
                using type = Arch;
            };

            template<class Arch, class... Archs, class... Args>
            struct join<Arch, arch_list<Archs...>, Args...> : join<typename Arch::template extend<Archs...>, Args...>
            {
            };
        }

        using all_x86 = arch_list<avx512, avx2, fma3, xop, fma4, avx, sse4_2, sse4_1, sse4a, sse3, sse2, sse>;
        using all_arm = arch_list<neon64, neon>;
        using all = typename detail::join<all_arm, all_x86, arch_list<scalar>>::type;

        using configured = typename detail::configured<all>::type;

        using x86 = typename detail::best<typename detail::configured<all_x86>::type>::type;
        using arm = typename detail::best<typename detail::configured<all_arm>::type>::type;

        using default_ = typename detail::best<typename detail::configured<arch_list<arm, x86, scalar>>::type>::type;

        // Generic, arch-dependent batch type.
        template <class T, class InstructionSet>
        using batch = typename InstructionSet::template batch<T>;
    }
}

#endif
