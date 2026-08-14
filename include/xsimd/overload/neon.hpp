
/****************************************************************************
 * Copyright (c) xsimd contributors                                         *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_OVERLOAD_NEON_HPP
#define XSIMD_OVERLOAD_NEON_HPP

#include "../config/xsimd_macros.hpp"
#include "../types/xsimd_batch.hpp"
#include "../utils/xsimd_type_traits.hpp"

#include <type_traits>

namespace xsimd::overload {

template<class T, class A>
XSIMD_INLINE constexpr bool vget_low_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vget_low_batch(batch<T, A> a) {
    static_assert(vget_low_is_supported<T, A>(), "vget_low unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vget_low_s8(a); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vget_low_u8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vget_low_s16(a); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vget_low_u16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vget_low_s32(a); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vget_low_u32(a); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vget_low_s64(a); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vget_low_u64(a); }
    else if constexpr(is_like_v<T, float>) { return vget_low_f32(a); }
    else if constexpr(is_like_v<T, double>) { return vget_low_f64(a); }
    else { static_assert(false, "unsupported type for vget_low"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vget_high_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vget_high_batch(batch<T, A> a) {
    static_assert(vget_high_is_supported<T, A>(), "vget_high unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vget_high_s8(a); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vget_high_u8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vget_high_s16(a); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vget_high_u16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vget_high_s32(a); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vget_high_u32(a); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vget_high_s64(a); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vget_high_u64(a); }
    else if constexpr(is_like_v<T, float>) { return vget_high_f32(a); }
    else if constexpr(is_like_v<T, double>) { return vget_high_f64(a); }
    else { static_assert(false, "unsupported type for vget_high"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vdupq_n_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vdupq_n_batch(T a) {
    static_assert(vdupq_n_is_supported<T, A>(), "vdupq_n unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vdupq_n_s8(a); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vdupq_n_u8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vdupq_n_s16(a); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vdupq_n_u16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vdupq_n_s32(a); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vdupq_n_u32(a); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vdupq_n_s64(a); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vdupq_n_u64(a); }
    else if constexpr(is_like_v<T, float>) { return vdupq_n_f32(a); }
    else if constexpr(is_like_v<T, double>) { return vdupq_n_f64(a); }
    else { static_assert(false, "unsupported type for vdupq_n"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vrev64q_is_supported() {
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vrev64q_batch(batch<T, A> a) {
    static_assert(vrev64q_is_supported<T, A>(), "vrev64q unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vrev64q_s8(a); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vrev64q_u8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vrev64q_s16(a); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vrev64q_u16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vrev64q_s32(a); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vrev64q_u32(a); }
    else if constexpr(is_like_v<T, float>) { return vrev64q_f32(a); }
    else { static_assert(false, "unsupported type for vrev64q"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vandq_is_supported() {
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t>;
}

template<class T, class A>
XSIMD_INLINE auto vandq_batch(batch<T, A> a, batch<T, A> b) {
    static_assert(vandq_is_supported<T, A>(), "vandq unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vandq_s8(a, b); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vandq_u8(a, b); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vandq_s16(a, b); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vandq_u16(a, b); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vandq_s32(a, b); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vandq_u32(a, b); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vandq_s64(a, b); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vandq_u64(a, b); }
    else { static_assert(false, "unsupported type for vandq"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vceqq_is_supported() {
    if constexpr(is_like_v<T, std::int64_t>) { return std::is_base_of_v<neon64, A>; }
    if constexpr(is_like_v<T, std::uint64_t>) { return std::is_base_of_v<neon64, A>; }
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vceqq_batch(batch<T, A> a, batch<T, A> b) {
    static_assert(vceqq_is_supported<T, A>(), "vceqq unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vceqq_s8(a, b); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vceqq_u8(a, b); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vceqq_s16(a, b); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vceqq_u16(a, b); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vceqq_s32(a, b); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vceqq_u32(a, b); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vceqq_s64(a, b); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vceqq_u64(a, b); }
    else if constexpr(is_like_v<T, float>) { return vceqq_f32(a, b); }
    else if constexpr(is_like_v<T, double>) { return vceqq_f64(a, b); }
    else { static_assert(false, "unsupported type for vceqq"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vaddq_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vaddq_batch(batch<T, A> a, batch<T, A> b) {
    static_assert(vaddq_is_supported<T, A>(), "vaddq unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vaddq_s8(a, b); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vaddq_u8(a, b); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vaddq_s16(a, b); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vaddq_u16(a, b); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vaddq_s32(a, b); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vaddq_u32(a, b); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vaddq_s64(a, b); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vaddq_u64(a, b); }
    else if constexpr(is_like_v<T, float>) { return vaddq_f32(a, b); }
    else if constexpr(is_like_v<T, double>) { return vaddq_f64(a, b); }
    else { static_assert(false, "unsupported type for vaddq"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vhaddq_is_supported() {
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t>;
}

template<class T, class A>
XSIMD_INLINE auto vhaddq_batch(batch<T, A> a, batch<T, A> b) {
    static_assert(vhaddq_is_supported<T, A>(), "vhaddq unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vhaddq_s8(a, b); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vhaddq_u8(a, b); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vhaddq_s16(a, b); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vhaddq_u16(a, b); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vhaddq_s32(a, b); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vhaddq_u32(a, b); }
    else { static_assert(false, "unsupported type for vhaddq"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vrhaddq_is_supported() {
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t>;
}

template<class T, class A>
XSIMD_INLINE auto vrhaddq_batch(batch<T, A> a, batch<T, A> b) {
    static_assert(vrhaddq_is_supported<T, A>(), "vrhaddq unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vrhaddq_s8(a, b); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vrhaddq_u8(a, b); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vrhaddq_s16(a, b); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vrhaddq_u16(a, b); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vrhaddq_s32(a, b); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vrhaddq_u32(a, b); }
    else { static_assert(false, "unsupported type for vrhaddq"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vqaddq_is_supported() {
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t>;
}

template<class T, class A>
XSIMD_INLINE auto vqaddq_batch(batch<T, A> a, batch<T, A> b) {
    static_assert(vqaddq_is_supported<T, A>(), "vqaddq unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vqaddq_s8(a, b); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vqaddq_u8(a, b); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vqaddq_s16(a, b); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vqaddq_u16(a, b); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vqaddq_s32(a, b); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vqaddq_u32(a, b); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vqaddq_s64(a, b); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vqaddq_u64(a, b); }
    else { static_assert(false, "unsupported type for vqaddq"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vnegq_is_supported() {
    if constexpr(is_like_v<T, std::int64_t>) { return std::is_base_of_v<neon64, A>; }
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::int16_t, std::int32_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vnegq_batch(batch<T, A> a) {
    static_assert(vnegq_is_supported<T, A>(), "vnegq unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vnegq_s8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vnegq_s16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vnegq_s32(a); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vnegq_s64(a); }
    else if constexpr(is_like_v<T, float>) { return vnegq_f32(a); }
    else if constexpr(is_like_v<T, double>) { return vnegq_f64(a); }
    else { static_assert(false, "unsupported type for vnegq"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vsubq_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vsubq_batch(batch<T, A> a, batch<T, A> b) {
    static_assert(vsubq_is_supported<T, A>(), "vsubq unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vsubq_s8(a, b); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vsubq_u8(a, b); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vsubq_s16(a, b); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vsubq_u16(a, b); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vsubq_s32(a, b); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vsubq_u32(a, b); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vsubq_s64(a, b); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vsubq_u64(a, b); }
    else if constexpr(is_like_v<T, float>) { return vsubq_f32(a, b); }
    else if constexpr(is_like_v<T, double>) { return vsubq_f64(a, b); }
    else { static_assert(false, "unsupported type for vsubq"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vqsubq_is_supported() {
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t>;
}

template<class T, class A>
XSIMD_INLINE auto vqsubq_batch(batch<T, A> a, batch<T, A> b) {
    static_assert(vqsubq_is_supported<T, A>(), "vqsubq unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vqsubq_s8(a, b); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vqsubq_u8(a, b); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vqsubq_s16(a, b); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vqsubq_u16(a, b); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vqsubq_s32(a, b); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vqsubq_u32(a, b); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vqsubq_s64(a, b); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vqsubq_u64(a, b); }
    else { static_assert(false, "unsupported type for vqsubq"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vmull_is_supported() {
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t>;
}

template<class T, class A>
XSIMD_INLINE auto vmull_batch(batch<T, A> a, batch<T, A> b) {
    static_assert(vmull_is_supported<T, A>(), "vmull unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vmull_s8(a, b); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vmull_u8(a, b); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vmull_s16(a, b); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vmull_u16(a, b); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vmull_s32(a, b); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vmull_u32(a, b); }
    else { static_assert(false, "unsupported type for vmull"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vmulq_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vmulq_batch(batch<T, A> a, batch<T, A> b) {
    static_assert(vmulq_is_supported<T, A>(), "vmulq unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vmulq_s8(a, b); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vmulq_u8(a, b); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vmulq_s16(a, b); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vmulq_u16(a, b); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vmulq_s32(a, b); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vmulq_u32(a, b); }
    else if constexpr(is_like_v<T, float>) { return vmulq_f32(a, b); }
    else if constexpr(is_like_v<T, double>) { return vmulq_f64(a, b); }
    else { static_assert(false, "unsupported type for vmulq"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vreinterpretq_s8_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vreinterpretq_s8_batch(batch<T, A> a) {
    static_assert(vreinterpretq_s8_is_supported<T, A>(), "vreinterpretq_s8 unsupported");
    if constexpr(is_like_v<T, std::uint8_t>) { return vreinterpretq_s8_u8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vreinterpretq_s8_s16(a); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vreinterpretq_s8_u16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vreinterpretq_s8_s32(a); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vreinterpretq_s8_u32(a); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vreinterpretq_s8_s64(a); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vreinterpretq_s8_u64(a); }
    else if constexpr(is_like_v<T, float>) { return vreinterpretq_s8_f32(a); }
    else if constexpr(is_like_v<T, double>) { return vreinterpretq_s8_f64(a); }
    else { static_assert(false, "unsupported type for vreinterpretq_s8"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vreinterpretq_u8_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vreinterpretq_u8_batch(batch<T, A> a) {
    static_assert(vreinterpretq_u8_is_supported<T, A>(), "vreinterpretq_u8 unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vreinterpretq_u8_s8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vreinterpretq_u8_s16(a); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vreinterpretq_u8_u16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vreinterpretq_u8_s32(a); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vreinterpretq_u8_u32(a); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vreinterpretq_u8_s64(a); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vreinterpretq_u8_u64(a); }
    else if constexpr(is_like_v<T, float>) { return vreinterpretq_u8_f32(a); }
    else if constexpr(is_like_v<T, double>) { return vreinterpretq_u8_f64(a); }
    else { static_assert(false, "unsupported type for vreinterpretq_u8"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vreinterpretq_s16_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vreinterpretq_s16_batch(batch<T, A> a) {
    static_assert(vreinterpretq_s16_is_supported<T, A>(), "vreinterpretq_s16 unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vreinterpretq_s16_s8(a); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vreinterpretq_s16_u8(a); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vreinterpretq_s16_u16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vreinterpretq_s16_s32(a); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vreinterpretq_s16_u32(a); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vreinterpretq_s16_s64(a); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vreinterpretq_s16_u64(a); }
    else if constexpr(is_like_v<T, float>) { return vreinterpretq_s16_f32(a); }
    else if constexpr(is_like_v<T, double>) { return vreinterpretq_s16_f64(a); }
    else { static_assert(false, "unsupported type for vreinterpretq_s16"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vreinterpretq_u16_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vreinterpretq_u16_batch(batch<T, A> a) {
    static_assert(vreinterpretq_u16_is_supported<T, A>(), "vreinterpretq_u16 unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vreinterpretq_u16_s8(a); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vreinterpretq_u16_u8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vreinterpretq_u16_s16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vreinterpretq_u16_s32(a); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vreinterpretq_u16_u32(a); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vreinterpretq_u16_s64(a); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vreinterpretq_u16_u64(a); }
    else if constexpr(is_like_v<T, float>) { return vreinterpretq_u16_f32(a); }
    else if constexpr(is_like_v<T, double>) { return vreinterpretq_u16_f64(a); }
    else { static_assert(false, "unsupported type for vreinterpretq_u16"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vreinterpretq_s32_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::uint32_t, std::int64_t, std::uint64_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vreinterpretq_s32_batch(batch<T, A> a) {
    static_assert(vreinterpretq_s32_is_supported<T, A>(), "vreinterpretq_s32 unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vreinterpretq_s32_s8(a); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vreinterpretq_s32_u8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vreinterpretq_s32_s16(a); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vreinterpretq_s32_u16(a); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vreinterpretq_s32_u32(a); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vreinterpretq_s32_s64(a); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vreinterpretq_s32_u64(a); }
    else if constexpr(is_like_v<T, float>) { return vreinterpretq_s32_f32(a); }
    else if constexpr(is_like_v<T, double>) { return vreinterpretq_s32_f64(a); }
    else { static_assert(false, "unsupported type for vreinterpretq_s32"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vreinterpretq_u32_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::int64_t, std::uint64_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vreinterpretq_u32_batch(batch<T, A> a) {
    static_assert(vreinterpretq_u32_is_supported<T, A>(), "vreinterpretq_u32 unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vreinterpretq_u32_s8(a); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vreinterpretq_u32_u8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vreinterpretq_u32_s16(a); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vreinterpretq_u32_u16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vreinterpretq_u32_s32(a); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vreinterpretq_u32_s64(a); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vreinterpretq_u32_u64(a); }
    else if constexpr(is_like_v<T, float>) { return vreinterpretq_u32_f32(a); }
    else if constexpr(is_like_v<T, double>) { return vreinterpretq_u32_f64(a); }
    else { static_assert(false, "unsupported type for vreinterpretq_u32"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vreinterpretq_s64_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::uint64_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vreinterpretq_s64_batch(batch<T, A> a) {
    static_assert(vreinterpretq_s64_is_supported<T, A>(), "vreinterpretq_s64 unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vreinterpretq_s64_s8(a); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vreinterpretq_s64_u8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vreinterpretq_s64_s16(a); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vreinterpretq_s64_u16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vreinterpretq_s64_s32(a); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vreinterpretq_s64_u32(a); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vreinterpretq_s64_u64(a); }
    else if constexpr(is_like_v<T, float>) { return vreinterpretq_s64_f32(a); }
    else if constexpr(is_like_v<T, double>) { return vreinterpretq_s64_f64(a); }
    else { static_assert(false, "unsupported type for vreinterpretq_s64"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vreinterpretq_u64_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, float>;
}

template<class T, class A>
XSIMD_INLINE auto vreinterpretq_u64_batch(batch<T, A> a) {
    static_assert(vreinterpretq_u64_is_supported<T, A>(), "vreinterpretq_u64 unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vreinterpretq_u64_s8(a); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vreinterpretq_u64_u8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vreinterpretq_u64_s16(a); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vreinterpretq_u64_u16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vreinterpretq_u64_s32(a); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vreinterpretq_u64_u32(a); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vreinterpretq_u64_s64(a); }
    else if constexpr(is_like_v<T, float>) { return vreinterpretq_u64_f32(a); }
    else if constexpr(is_like_v<T, double>) { return vreinterpretq_u64_f64(a); }
    else { static_assert(false, "unsupported type for vreinterpretq_u64"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vreinterpretq_f32_is_supported() {
    if constexpr(is_like_v<T, double>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t, std::uint64_t>;
}

template<class T, class A>
XSIMD_INLINE auto vreinterpretq_f32_batch(batch<T, A> a) {
    static_assert(vreinterpretq_f32_is_supported<T, A>(), "vreinterpretq_f32 unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vreinterpretq_f32_s8(a); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vreinterpretq_f32_u8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vreinterpretq_f32_s16(a); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vreinterpretq_f32_u16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vreinterpretq_f32_s32(a); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vreinterpretq_f32_u32(a); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vreinterpretq_f32_s64(a); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vreinterpretq_f32_u64(a); }
    else if constexpr(is_like_v<T, double>) { return vreinterpretq_f32_f64(a); }
    else { static_assert(false, "unsupported type for vreinterpretq_f32"); }
}

template<class T, class A>
XSIMD_INLINE constexpr bool vreinterpretq_f64_is_supported() {
    if constexpr(is_like_v<T, std::int8_t>) { return std::is_base_of_v<neon64, A>; }
    if constexpr(is_like_v<T, std::uint8_t>) { return std::is_base_of_v<neon64, A>; }
    if constexpr(is_like_v<T, std::int16_t>) { return std::is_base_of_v<neon64, A>; }
    if constexpr(is_like_v<T, std::uint16_t>) { return std::is_base_of_v<neon64, A>; }
    if constexpr(is_like_v<T, std::int32_t>) { return std::is_base_of_v<neon64, A>; }
    if constexpr(is_like_v<T, std::uint32_t>) { return std::is_base_of_v<neon64, A>; }
    if constexpr(is_like_v<T, std::int64_t>) { return std::is_base_of_v<neon64, A>; }
    if constexpr(is_like_v<T, float>) { return std::is_base_of_v<neon64, A>; }
    return is_like_any_v<T, std::uint64_t>;
}

template<class T, class A>
XSIMD_INLINE auto vreinterpretq_f64_batch(batch<T, A> a) {
    static_assert(vreinterpretq_f64_is_supported<T, A>(), "vreinterpretq_f64 unsupported");
    if constexpr(is_like_v<T, std::int8_t>) { return vreinterpretq_f64_s8(a); }
    else if constexpr(is_like_v<T, std::uint8_t>) { return vreinterpretq_f64_u8(a); }
    else if constexpr(is_like_v<T, std::int16_t>) { return vreinterpretq_f64_s16(a); }
    else if constexpr(is_like_v<T, std::uint16_t>) { return vreinterpretq_f64_u16(a); }
    else if constexpr(is_like_v<T, std::int32_t>) { return vreinterpretq_f64_s32(a); }
    else if constexpr(is_like_v<T, std::uint32_t>) { return vreinterpretq_f64_u32(a); }
    else if constexpr(is_like_v<T, std::int64_t>) { return vreinterpretq_f64_s64(a); }
    else if constexpr(is_like_v<T, std::uint64_t>) { return vreinterpretq_f64_u64(a); }
    else if constexpr(is_like_v<T, float>) { return vreinterpretq_f64_f32(a); }
    else { static_assert(false, "unsupported type for vreinterpretq_f64"); }
}

}  // namespace xsimd::overload

#endif  // XSIMD_OVERLOAD_NEON_HPP
