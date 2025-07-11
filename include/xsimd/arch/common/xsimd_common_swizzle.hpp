/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 * Copyright (c) Marco Barbone                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software.*
 ****************************************************************************/
#ifndef XSIMD_COMMON_SWIZZLE_HPP
#define XSIMD_COMMON_SWIZZLE_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace xsimd
{
    template <typename T, class A, T... Values>
    struct batch_constant;

    namespace kernel
    {
        namespace detail
        {
            // ────────────────────────────────────────────────────────────────────────
            //  get_at<I,Values...> → the I-th element of the pack
            template <typename T, std::size_t I, T V0, T... Vs>
            struct get_at
            {
                static constexpr T value = get_at<T, I - 1, Vs...>::value;
            };
            template <typename T, T V0, T... Vs>
            struct get_at<T, 0, V0, Vs...>
            {
                static constexpr T value = V0;
            };

            // ────────────────────────────────────────────────────────────────────────
            //  1) identity_impl
            template <std::size_t /*I*/, typename T>
            XSIMD_INLINE constexpr bool identity_impl() noexcept { return true; }
            template <std::size_t I, typename T, T V0, T... Vs>
            XSIMD_INLINE constexpr bool identity_impl() noexcept
            {
                return V0 == static_cast<T>(I)
                    && identity_impl<I + 1, T, Vs...>();
            }

            // ────────────────────────────────────────────────────────────────────────
            //  2) bitmask_impl
            template <std::size_t /*I*/, std::size_t /*N*/, typename T>
            XSIMD_INLINE constexpr std::uint32_t bitmask_impl() noexcept { return 0u; }
            template <std::size_t I, std::size_t N, typename T, T V0, T... Vs>
            XSIMD_INLINE constexpr std::uint32_t bitmask_impl() noexcept
            {
                return (1u << (static_cast<std::uint32_t>(V0) & (N - 1)))
                    | bitmask_impl<I + 1, N, T, Vs...>();
            }

            // ────────────────────────────────────────────────────────────────────────
            //  3) dup_lo_impl
            template <std::size_t I, std::size_t N, typename T,
                      T... Vs, typename std::enable_if<I == N / 2, int>::type = 0>
            XSIMD_INLINE constexpr bool dup_lo_impl() noexcept { return true; }

            template <std::size_t I, std::size_t N, typename T,
                      T... Vs, typename std::enable_if<(I < N / 2), int>::type = 0>
            XSIMD_INLINE constexpr bool dup_lo_impl() noexcept
            {
                return get_at<T, I, Vs...>::value < static_cast<T>(N / 2)
                    && get_at<T, I + N / 2, Vs...>::value == get_at<T, I, Vs...>::value
                    && dup_lo_impl<I + 1, N, T, Vs...>();
            }

            // ────────────────────────────────────────────────────────────────────────
            //  4) dup_hi_impl
            template <std::size_t I, std::size_t N, typename T,
                      T... Vs, typename std::enable_if<I == N / 2, int>::type = 0>
            XSIMD_INLINE constexpr bool dup_hi_impl() noexcept { return true; }

            template <std::size_t I, std::size_t N, typename T,
                      T... Vs, typename std::enable_if<(I < N / 2), int>::type = 0>
            XSIMD_INLINE constexpr bool dup_hi_impl() noexcept
            {
                return get_at<T, I, Vs...>::value >= static_cast<T>(N / 2)
                    && get_at<T, I, Vs...>::value < static_cast<T>(N)
                    && get_at<T, I + N / 2, Vs...>::value == get_at<T, I, Vs...>::value
                    && dup_hi_impl<I + 1, N, T, Vs...>();
            }

            // ────────────────────────────────────────────────────────────────────────
            //  1) helper to get the I-th value from the Vs pack
            template <std::size_t I, uint32_t Head, uint32_t... Tail>
            struct get_nth_value
            {
                static constexpr uint32_t value = get_nth_value<I - 1, Tail...>::value;
            };
            template <uint32_t Head, uint32_t... Tail>
            struct get_nth_value<0, Head, Tail...>
            {
                static constexpr uint32_t value = Head;
            };

            // ────────────────────────────────────────────────────────────────────────
            //  2) recursive cross‐lane test: true if any output‐lane i pulls from the opposite half
            template <std::size_t I,
                      std::size_t N,
                      std::size_t H,
                      uint32_t... Vs>
            struct cross_impl
            {
                // does element I cross? (i.e. i<H but V>=H) or (i>=H but V<H)
                static constexpr uint32_t Vi = get_nth_value<I, Vs...>::value;
                static constexpr bool curr = (I < H ? (Vi >= H) : (Vi < H));
                static constexpr bool next = cross_impl<I + 1, N, H, Vs...>::value;
                static constexpr bool value = curr || next;
            };
            template <std::size_t N, std::size_t H, uint32_t... Vs>
            struct cross_impl<N, N, H, Vs...>
            {
                static constexpr bool value = false;
            };
            template <std::size_t I, std::size_t N, typename T,
                      T... Vs>
            XSIMD_INLINE constexpr bool no_duplicates_impl() noexcept
            {
                // build the bitmask of (Vs & (N-1)) across all lanes
                return detail::bitmask_impl<0, N, T, Vs...>() == ((1u << N) - 1u);
            }
            template <uint32_t... Vs>
            XSIMD_INLINE constexpr bool no_duplicates_v() noexcept
            {
                // forward to your existing no_duplicates_impl
                return no_duplicates_impl<0, sizeof...(Vs), uint32_t, Vs...>();
            }
            template <uint32_t... Vs>
            XSIMD_INLINE constexpr bool is_cross_lane() noexcept
            {
                static_assert(sizeof...(Vs) >= 1, "Need at least one lane");
                return cross_impl<0, sizeof...(Vs), sizeof...(Vs) / 2, Vs...>::value;
            }
            template <typename T, T... Vs>
            XSIMD_INLINE constexpr bool is_identity() noexcept { return detail::identity_impl<0, T, Vs...>(); }
            template <typename T, T... Vs>
            XSIMD_INLINE constexpr bool is_all_different() noexcept
            {
                return detail::bitmask_impl<0, sizeof...(Vs), T, Vs...>() == ((1u << sizeof...(Vs)) - 1);
            }

            template <typename T, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_lo() noexcept { return detail::dup_lo_impl<0, sizeof...(Vs), T, Vs...>(); }
            template <typename T, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_hi() noexcept { return detail::dup_hi_impl<0, sizeof...(Vs), T, Vs...>(); }
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_identity(batch_constant<T, A, Vs...>) noexcept { return is_identity<T, Vs...>(); }
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_all_different(batch_constant<T, A, Vs...>) noexcept { return is_all_different<T, Vs...>(); }
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_lo(batch_constant<T, A, Vs...>) noexcept { return is_dup_lo<T, Vs...>(); }
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_hi(batch_constant<T, A, Vs...>) noexcept { return is_dup_hi<T, Vs...>(); }
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_cross_lane(batch_constant<T, A, Vs...>) noexcept { return detail::is_cross_lane<Vs...>(); }
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool no_duplicates(batch_constant<T, A, Vs...>) noexcept { return no_duplicates_impl<0, sizeof...(Vs), T, Vs...>(); }
            // ────────────────────────────────────────────────────────────────────────
            //  compile-time tests (identity, all-different, dup-lo, dup-hi)
            //  8-lane identity
            static_assert(is_identity<std::uint32_t, 0, 1, 2, 3, 4, 5, 6, 7>(), "identity failed");
            // 8-lane reverse is all-different but not identity
            static_assert(is_all_different<std::uint32_t, 7, 6, 5, 4, 3, 2, 1, 0>(), "all-diff failed");
            static_assert(!is_identity<std::uint32_t, 7, 6, 5, 4, 3, 2, 1, 0>(), "identity on reverse");
            // 8-lane dup-lo (repeat 0..3 twice)
            static_assert(is_dup_lo<std::uint32_t, 0, 1, 2, 3, 0, 1, 2, 3>(), "dup_lo failed");
            static_assert(!is_dup_hi<std::uint32_t, 0, 1, 2, 3, 0, 1, 2, 3>(), "dup_hi on dup_lo");
            // 8-lane dup-hi (repeat 4..7 twice)
            static_assert(is_dup_hi<std::uint32_t, 4, 5, 6, 7, 4, 5, 6, 7>(), "dup_hi failed");
            static_assert(!is_dup_lo<std::uint32_t, 4, 5, 6, 7, 4, 5, 6, 7>(), "dup_lo on dup_hi");
            // ────────────────────────────────────────────────────────────────────────
            //  4-lane identity
            static_assert(is_identity<std::uint32_t, 0, 1, 2, 3>(), "4-lane identity failed");
            // 4-lane reverse all-different but not identity
            static_assert(is_all_different<std::uint32_t, 3, 2, 1, 0>(), "4-lane all-diff failed");
            static_assert(!is_identity<std::uint32_t, 3, 2, 1, 0>(), "4-lane identity on reverse");
            // 4-lane dup-lo (repeat 0..1 twice)
            static_assert(is_dup_lo<std::uint32_t, 0, 1, 0, 1>(), "4-lane dup_lo failed");
            static_assert(!is_dup_hi<std::uint32_t, 0, 1, 0, 1>(), "4-lane dup_hi on dup_lo");
            // 4-lane dup-hi (repeat 2..3 twice)
            static_assert(is_dup_hi<std::uint32_t, 2, 3, 2, 3>(), "4-lane dup_hi failed");
            static_assert(!is_dup_lo<std::uint32_t, 2, 3, 2, 3>(), "4-lane dup_lo on dup_hi");

            static_assert(is_cross_lane<0, 1, 0, 1>(), "dup-lo only → crossing");
            static_assert(is_cross_lane<2, 3, 2, 3>(), "dup-hi only → crossing");
            static_assert(is_cross_lane<0, 3, 3, 3>(), "one low + rest high → crossing");
            static_assert(!is_cross_lane<1, 0, 2, 3>(), "mixed low/high → no crossing");
            static_assert(!is_cross_lane<0, 1, 2, 3>(), "mixed low/high → no crossing");

            static_assert(no_duplicates_v<0, 1, 2, 3>(), "N=4: [0,1,2,3] → distinct");
            static_assert(!no_duplicates_v<0, 1, 2, 2>(), "N=4: [0,1,2,2] → dup");

            static_assert(no_duplicates_v<0, 1, 2, 3, 4, 5, 6, 7>(), "N=8: [0..7] → distinct");
            static_assert(!no_duplicates_v<0, 1, 2, 3, 4, 5, 6, 0>(), "N=8: last repeats 0");

        } // namespace detail
    } // namespace kernel
} // namespace xsimd

#endif // XSIMD_COMMON_SWIZZLE_HPP
