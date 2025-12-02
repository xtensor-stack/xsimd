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

#include "../../config/xsimd_inline.hpp"

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
            // identity_impl
            template <std::size_t /*I*/, typename T>
            XSIMD_INLINE constexpr bool identity_impl() noexcept { return true; }
            template <std::size_t I, typename T, T V0, T... Vs>
            XSIMD_INLINE constexpr bool identity_impl() noexcept
            {
                return V0 == static_cast<T>(I)
                    && identity_impl<I + 1, T, Vs...>();
            }

            // ────────────────────────────────────────────────────────────────────────
            // dup_lo_impl
            template <std::size_t I, std::size_t N, typename T,
                      T... Vs, std::enable_if_t<I == N / 2, int> = 0>
            XSIMD_INLINE constexpr bool dup_lo_impl() noexcept { return true; }

            template <std::size_t I, std::size_t N, typename T,
                      T... Vs, std::enable_if_t<(I < N / 2), int> = 0>
            XSIMD_INLINE constexpr bool dup_lo_impl() noexcept
            {
                return get_at<T, I, Vs...>::value < static_cast<T>(N / 2)
                    && get_at<T, I + N / 2, Vs...>::value == get_at<T, I, Vs...>::value
                    && dup_lo_impl<I + 1, N, T, Vs...>();
            }

            // ────────────────────────────────────────────────────────────────────────
            // dup_hi_impl
            template <std::size_t I, std::size_t N, typename T,
                      T... Vs, std::enable_if_t<I == N / 2, int> = 0>
            XSIMD_INLINE constexpr bool dup_hi_impl() noexcept { return true; }

            template <std::size_t I, std::size_t N, typename T,
                      T... Vs, std::enable_if_t<(I < N / 2), int> = 0>
            XSIMD_INLINE constexpr bool dup_hi_impl() noexcept
            {
                return get_at<T, I, Vs...>::value >= static_cast<T>(N / 2)
                    && get_at<T, I, Vs...>::value < static_cast<T>(N)
                    && get_at<T, I + N / 2, Vs...>::value == get_at<T, I, Vs...>::value
                    && dup_hi_impl<I + 1, N, T, Vs...>();
            }

            // ────────────────────────────────────────────────────────────────────────
            // only_from_lo
            template <typename T, T Size, T First, T... Vals>
            struct only_from_lo_impl;

            template <typename T, T Size, T Last>
            struct only_from_lo_impl<T, Size, Last>
            {
                static constexpr bool value = (Last < (Size / 2));
            };

            template <typename T, T Size, T First, T... Vals>
            struct only_from_lo_impl
            {
                static constexpr bool value = (First < (Size / 2)) && only_from_lo_impl<T, Size, Vals...>::value;
            };

            template <typename T, T... Vals>
            constexpr bool is_only_from_lo()
            {
                return only_from_lo_impl<T, sizeof...(Vals), Vals...>::value;
            };

            // ────────────────────────────────────────────────────────────────────────
            // only_from_hi
            template <typename T, T Size, T First, T... Vals>
            struct only_from_hi_impl;

            template <typename T, T Size, T Last>
            struct only_from_hi_impl<T, Size, Last>
            {
                static constexpr bool value = (Last >= (Size / 2));
            };

            template <typename T, T Size, T First, T... Vals>
            struct only_from_hi_impl
            {
                static constexpr bool value = (First >= (Size / 2)) && only_from_hi_impl<T, Size, Vals...>::value;
            };

            template <typename T, T... Vals>
            constexpr bool is_only_from_hi()
            {
                return only_from_hi_impl<T, sizeof...(Vals), Vals...>::value;
            };

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
            template <uint32_t... Vs>
            XSIMD_INLINE constexpr bool is_cross_lane() noexcept
            {
                static_assert(sizeof...(Vs) >= 1, "Need at least one lane");
                return cross_impl<0, sizeof...(Vs), sizeof...(Vs) / 2, Vs...>::value;
            }

            template <typename T, T... Vs>
            XSIMD_INLINE constexpr bool is_identity() noexcept { return detail::identity_impl<0, T, Vs...>(); }
            template <typename T, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_lo() noexcept { return detail::dup_lo_impl<0, sizeof...(Vs), T, Vs...>(); }
            template <typename T, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_hi() noexcept { return detail::dup_hi_impl<0, sizeof...(Vs), T, Vs...>(); }
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_identity(batch_constant<T, A, Vs...>) noexcept { return is_identity<T, Vs...>(); }
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_lo(batch_constant<T, A, Vs...>) noexcept { return is_dup_lo<T, Vs...>(); }
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_hi(batch_constant<T, A, Vs...>) noexcept { return is_dup_hi<T, Vs...>(); }
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_only_from_lo(batch_constant<T, A, Vs...>) noexcept { return detail::is_only_from_lo<T, Vs...>(); }
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_only_from_hi(batch_constant<T, A, Vs...>) noexcept { return detail::is_only_from_hi<T, Vs...>(); }
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_cross_lane(batch_constant<T, A, Vs...>) noexcept { return detail::is_cross_lane<Vs...>(); }

        } // namespace detail
    } // namespace kernel
} // namespace xsimd

#endif // XSIMD_COMMON_SWIZZLE_HPP
