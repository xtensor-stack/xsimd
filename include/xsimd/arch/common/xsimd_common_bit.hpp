/****************************************************************
 * Partial backport of `__cpp_lib_bitops == 201907L` from C++20 *
 ****************************************************************/

#ifndef XSIMD_BIT_HPP
#define XSIMD_BIT_HPP

#include "../../config/xsimd_config.hpp"

#include <climits>
#include <cstddef>
#include <type_traits>

#if XSIMD_CPP_VERSION >= 202002L
#include <version>
#endif

#if XSIMD_CPP_VERSION >= 202002L && __cpp_lib_bitops >= 201907L
#include <bit>
#define XSIMD_HAS_STD_BITOPS 1
#endif

#ifdef __has_builtin
#define XSIMD_HAS_BUILTIN(x) __has_builtin(x)
#else
#define XSIMD_HAS_BUILTIN(x) 0
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace xsimd
{
    namespace detail
    {
        // Lane value made of \c byte repeated across every byte of \c T.
        template <class T>
        XSIMD_INLINE constexpr T repeat_pattern(unsigned char byte) noexcept
        {
            T out = 0;
            for (std::size_t i = 0; i < sizeof(T); ++i)
                out = T((out << CHAR_BIT) | byte);
            return out;
        }

        // Portable word-parallel popcount. Bits fold to per-byte counts, then
        // a single multiply sums the byte counts into the top byte. W is at
        // least unsigned int, otherwise the final multiply loses the sum for
        // narrow T.
        // https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
        template <class T>
        XSIMD_INLINE constexpr int popcount_swar(T v) noexcept
        {
            static_assert(std::is_unsigned_v<T>, "popcount requires an unsigned integral type");
            using W = std::conditional_t<(sizeof(T) < sizeof(unsigned int)), unsigned int, T>;
            W x = W(v);
            x = x - ((x >> 1) & repeat_pattern<W>(0x55));
            x = (x & repeat_pattern<W>(0x33)) + ((x >> 2) & repeat_pattern<W>(0x33));
            x = (x + (x >> 4)) & repeat_pattern<W>(0x0f);
            return int((x * W(~W(0) / 255)) >> ((sizeof(W) - 1) * CHAR_BIT));
        }

        // Dispatch order, first match wins: SWAR for GCC on x86 without
        // POPCNT, std::popcount (C++20), __builtin_popcountg (GCC 14,
        // Clang 19), sized builtins, MSVC intrinsics, SWAR. The first rung
        // beats std::popcount because GCC lowers it, like every builtin, to
        // a libgcc call that is slower than the inline fold. The builtins
        // fold to constant expressions, the MSVC intrinsics do not.
#if defined(_MSC_VER) && !defined(__clang__) && !defined(XSIMD_HAS_STD_BITOPS)
        template <class T>
        XSIMD_INLINE int popcount(T x) noexcept
#else
        template <class T>
        XSIMD_INLINE constexpr int popcount(T x) noexcept
#endif
        {
            static_assert(std::is_unsigned_v<T>, "popcount requires an unsigned integral type");
#if defined(__GNUC__) && !defined(__clang__) && (defined(__i386__) || defined(__x86_64__)) && !defined(__POPCNT__)
            return popcount_swar(x);
#elif defined(XSIMD_HAS_STD_BITOPS)
            return std::popcount(x);
#elif XSIMD_HAS_BUILTIN(__builtin_popcountg)
            return __builtin_popcountg(x);
#elif XSIMD_HAS_BUILTIN(__builtin_popcount) && XSIMD_HAS_BUILTIN(__builtin_popcountll)
            if constexpr (sizeof(T) <= 4)
            {
                return __builtin_popcount(x);
            }
            else
            {
                return __builtin_popcountll(x);
            }
#elif defined(_MSC_VER)
            if constexpr (sizeof(T) == 2)
            {
                return __popcnt16(x);
            }
            else if constexpr (sizeof(T) == 1 || sizeof(T) == 4)
            {
                return __popcnt(x);
            }
            else
            {
                // sizeof(T) == 8
#ifdef _M_X64
                return (int)__popcnt64(x);
#else
                return (int)(__popcnt((unsigned int)x) + __popcnt((unsigned int)(x >> 32)));
#endif
            }
#else
            return popcount_swar(x);
#endif
        }

#ifdef XSIMD_HAS_STD_BITOPS
        using std::countl_one;
        using std::countl_zero;
        using std::countr_one;
        using std::countr_zero;
#else
        template <class T, class = std::enable_if_t<std::is_unsigned_v<T>>>
        XSIMD_INLINE int countl_zero(T x) noexcept
        {
#if XSIMD_HAS_BUILTIN(__builtin_clzg)
            return __builtin_clzg(x, (int)(sizeof(T) * CHAR_BIT));
#else
            if (x == 0)
                return sizeof(T) * CHAR_BIT;

            if constexpr (sizeof(T) <= 4)
            {
#if XSIMD_HAS_BUILTIN(__builtin_clz)
                return __builtin_clz((unsigned int)x) - (4 - sizeof(T)) * CHAR_BIT;
#elif defined(_MSC_VER)
                unsigned long index;
                _BitScanReverse(&index, (unsigned long)x);
                return sizeof(T) * CHAR_BIT - index - 1;
#else
                x |= x >> 1;
                x |= x >> 2;
                x |= x >> 4;
                if constexpr (sizeof(T) >= 2)
                {
                    x |= x >> 8;
                }
                if constexpr (sizeof(T) >= 4)
                {
                    x |= x >> 16;
                }
                return sizeof(T) * CHAR_BIT - popcount(x);
#endif
            }
            else
            {
                // sizeof(T) == 8
#if XSIMD_HAS_BUILTIN(__builtin_clzll)
                return __builtin_clzll((unsigned long long)x);
#elif defined(_MSC_VER) && defined(_M_X64)
                unsigned long index;
                _BitScanReverse64(&index, (unsigned long long)x);
                return sizeof(T) * CHAR_BIT - index - 1;
#else
                x |= x >> 1;
                x |= x >> 2;
                x |= x >> 4;
                x |= x >> 8;
                x |= x >> 16;
                x |= x >> 32;
                return sizeof(T) * CHAR_BIT - popcount(x);
#endif
            }
#endif
        }

        template <class T, class = std::enable_if_t<std::is_unsigned_v<T>>>
        XSIMD_INLINE int countl_one(T x) noexcept
        {
            return countl_zero(T(~x));
        }

        template <class T, class = std::enable_if_t<std::is_unsigned_v<T>>>
        XSIMD_INLINE int countr_zero(T x) noexcept
        {
#if XSIMD_HAS_BUILTIN(__builtin_ctzg)
            return __builtin_ctzg(x, (int)(sizeof(T) * CHAR_BIT));
#else
            if (x == 0)
                return sizeof(T) * CHAR_BIT;

            if constexpr (sizeof(T) <= 4)
            {
#if XSIMD_HAS_BUILTIN(__builtin_ctz)
                return __builtin_ctz((unsigned int)x);
#elif defined(_MSC_VER)
                unsigned long index;
                _BitScanForward(&index, (unsigned long)x);
                return index;
#endif
            }
            else
            {
                // sizeof(T) == 8
#if XSIMD_HAS_BUILTIN(__builtin_ctzll)
                return __builtin_ctzll((unsigned long long)x);
#elif defined(_MSC_VER) && defined(_M_X64)
                unsigned long index;
                _BitScanForward64(&index, (unsigned long long)x);
                return index;
#endif
            }

            // https://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightMultLookup
            return popcount((T)((x & -x) - 1));
#endif
        }

        template <class T, class = std::enable_if_t<std::is_unsigned_v<T>>>
        XSIMD_INLINE int countr_one(T x) noexcept
        {
            return countr_zero(T(~x));
        }
#endif
    }
}

#undef XSIMD_HAS_STD_BITOPS
#undef XSIMD_HAS_BUILTIN

#endif
