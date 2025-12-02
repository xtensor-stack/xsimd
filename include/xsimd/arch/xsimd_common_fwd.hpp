/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 * Copyright (c) Marco Barbone                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_COMMON_FWD_HPP
#define XSIMD_COMMON_FWD_HPP

#include <cstdint>
#include <type_traits>

namespace xsimd
{
    // Minimal forward declarations used in this header
    template <class T, class A>
    class batch;
    template <class T, class A>
    class batch_bool;
    template <class T, class A, T... Vs>
    struct batch_constant;
    template <class T, class A, bool... Vs>
    struct batch_bool_constant;
    template <class T>
    struct convert;
    template <class A>
    struct requires_arch;
    struct aligned_mode;
    struct unaligned_mode;

    namespace types
    {
        template <typename T, class A>
        struct has_simd_register;
    }

    namespace kernel
    {
        // forward declaration
        template <class A, class T, class = std::enable_if_t<std::is_integral<T>::value>>
        XSIMD_INLINE batch<T, A> abs(batch<T, A> const& self, requires_arch<common>) noexcept;
        template <class A, class T, class = std::enable_if_t<std::is_integral<T>::value>>
        XSIMD_INLINE batch<T, A> bitwise_lshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <size_t shift, class A, class T, class = std::enable_if_t<std::is_integral<T>::value>>
        XSIMD_INLINE batch<T, A> bitwise_lshift(batch<T, A> const& self, requires_arch<common>) noexcept;
        template <class A, class T, class = std::enable_if_t<std::is_integral<T>::value>>
        XSIMD_INLINE batch<T, A> bitwise_rshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <size_t shift, class A, class T, class = std::enable_if_t<std::is_integral<T>::value>>
        XSIMD_INLINE batch<T, A> bitwise_rshift(batch<T, A> const& self, requires_arch<common>) noexcept;
        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = std::enable_if_t<std::is_integral<T>::value>>
        XSIMD_INLINE batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = std::enable_if_t<std::is_integral<T>::value>>
        XSIMD_INLINE batch<T, A> sadd(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = std::enable_if_t<std::is_integral<T>::value>>
        XSIMD_INLINE batch<T, A> ssub(batch<T, A> const& self, batch<T, A> const& other, requires_arch<common>) noexcept;
        template <class A, class T, class = std::enable_if_t<std::is_scalar<T>::value>>
        XSIMD_INLINE T reduce_add(batch<T, A> const& self, requires_arch<common>) noexcept;
        template <class A, class T, class = std::enable_if_t<std::is_scalar<T>::value>>
        XSIMD_INLINE T reduce_mul(batch<T, A> const& self, requires_arch<common>) noexcept;
        template <class A, class T, class STy>
        XSIMD_INLINE batch<T, A> rotl(batch<T, A> const& self, STy other, requires_arch<common>) noexcept;
        template <size_t count, class A, class T>
        XSIMD_INLINE batch<T, A> rotl(batch<T, A> const& self, requires_arch<common>) noexcept;
        template <class A, class T, class STy>
        XSIMD_INLINE batch<T, A> rotr(batch<T, A> const& self, STy other, requires_arch<common>) noexcept;
        template <size_t count, class A, class T>
        XSIMD_INLINE batch<T, A> rotr(batch<T, A> const& self, requires_arch<common>) noexcept;
        template <class A, class T>
        XSIMD_INLINE batch<T, A> load(T const* mem, aligned_mode, requires_arch<A>) noexcept;
        template <class A, class T>
        XSIMD_INLINE batch<T, A> load(T const* mem, unaligned_mode, requires_arch<A>) noexcept;
        template <class A, class T_in, class T_out, bool... Values, class alignment>
        XSIMD_INLINE batch<T_out, A> load_masked(T_in const* mem, batch_bool_constant<T_out, A, Values...> mask, convert<T_out>, alignment, requires_arch<common>) noexcept;
        template <class A, class T_in, class T_out, bool... Values, class alignment>
        XSIMD_INLINE void store_masked(T_out* mem, batch<T_in, A> const& src, batch_bool_constant<T_in, A, Values...> mask, alignment, requires_arch<common>) noexcept;
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<int32_t, A> load_masked(int32_t const* mem, batch_bool_constant<int32_t, A, Values...> mask, convert<int32_t>, Mode, requires_arch<A>) noexcept;
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<uint32_t, A> load_masked(uint32_t const* mem, batch_bool_constant<uint32_t, A, Values...> mask, convert<uint32_t>, Mode, requires_arch<A>) noexcept;
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE std::enable_if_t<types::has_simd_register<double, A>::value, batch<int64_t, A>> load_masked(int64_t const*, batch_bool_constant<int64_t, A, Values...>, convert<int64_t>, Mode, requires_arch<A>) noexcept;
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE std::enable_if_t<types::has_simd_register<double, A>::value, batch<uint64_t, A>> load_masked(uint64_t const*, batch_bool_constant<uint64_t, A, Values...>, convert<uint64_t>, Mode, requires_arch<A>) noexcept;
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(int32_t* mem, batch<int32_t, A> const& src, batch_bool_constant<int32_t, A, Values...> mask, Mode, requires_arch<A>) noexcept;
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(uint32_t* mem, batch<uint32_t, A> const& src, batch_bool_constant<uint32_t, A, Values...> mask, Mode, requires_arch<A>) noexcept;
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE std::enable_if_t<types::has_simd_register<double, A>::value> store_masked(int64_t*, batch<int64_t, A> const&, batch_bool_constant<int64_t, A, Values...>, Mode, requires_arch<A>) noexcept;
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE std::enable_if_t<types::has_simd_register<double, A>::value> store_masked(uint64_t*, batch<uint64_t, A> const&, batch_bool_constant<uint64_t, A, Values...>, Mode, requires_arch<A>) noexcept;

        // Forward declarations for pack-level helpers
        namespace detail
        {
            template <typename T, T... Vs>
            XSIMD_INLINE constexpr bool is_identity() noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_identity(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_lo(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_dup_hi(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_cross_lane(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_only_from_lo(batch_constant<T, A, Vs...>) noexcept;
            template <typename T, class A, T... Vs>
            XSIMD_INLINE constexpr bool is_only_from_hi(batch_constant<T, A, Vs...>) noexcept;

        }
    }
}

#endif
