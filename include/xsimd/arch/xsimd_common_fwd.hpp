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

#ifndef XSIMD_COMMON_FWD_HPP
#define XSIMD_COMMON_FWD_HPP

#include "../types/xsimd_batch_constant.hpp"

#include <type_traits>

namespace xsimd
{
    namespace kernel
    {
        // forward declaration
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> abs(batch<T, A> const& self, requires_arch<common>) noexcept;
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> bitwise_lshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> bitwise_rshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> sadd(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> ssub(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE T hadd(batch<T, A> const& self, requires_arch<common>) noexcept;
        // Forward declarations for pack-level helpers
        namespace detail
        {
            template <typename T, T... Vs>
            XSIMD_INLINE constexpr bool is_identity() noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_identity(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_all_different(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_lo(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_hi(batch_constant<T, A, Vs...>) noexcept;
            template <uint32_t... Values>
            XSIMD_INLINE constexpr std::uint32_t shuffle() noexcept;
            template <uint32_t... Values>
            XSIMD_INLINE constexpr std::uint32_t mod_shuffle() noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_cross_lane(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool no_duplicates(batch_constant<T, A, Vs...>) noexcept;

        }
    }
}

#endif
