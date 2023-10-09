/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 * Copyright (c) Anutosh Bhat                                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_WASM_HPP
#define XSIMD_WASM_HPP

#include <type_traits>

#include "../types/xsimd_wasm_register.hpp"

namespace xsimd
{

    namespace kernel
    {
        using namespace types;

        // abs
        template <class A, class T, typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, void>::type>
        inline batch<T, A> abs(batch<T, A> const& self, requires_arch<wasm>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
            {
                return wasm_i8x16_abs(self);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
            {
                return wasm_i16x8_abs(self);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                return wasm_i32x4_abs(self);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
            {
                return wasm_i64x2_abs(self);
            }
            else
            {
                assert(false && "unsupported arch/op combination");
                return {};
            }
        }

        template <class A>
        inline batch<float, A> abs(batch<float, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_abs(self);
        }

        template <class A>
        inline batch<double, A> abs(batch<double, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_abs(self);
        }

        // add
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> add(batch<T, A> const& self, batch<T, A> const& other, requires_arch<wasm>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
            {
                return wasm_i8x16_add(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
            {
                return wasm_i16x8_add(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                return wasm_i32x4_add(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
            {
                return wasm_i64x2_add(self, other);
            }
            else
            {
                assert(false && "unsupported arch/op combination");
                return {};
            }
        }

        template <class A>
        inline batch<float, A> add(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_add(self, other);
        }

        template <class A>
        inline batch<double, A> add(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_add(self, other);
        }

        // all
        template <class A>
        inline bool all(batch_bool<float, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_i32x4_bitmask(self) == 0x0F;
        }
        template <class A>
        inline bool all(batch_bool<double, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_i64x2_bitmask(self) == 0x03;
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline bool all(batch_bool<T, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_i8x16_bitmask(self) == 0xFFFF;
        }

        // any
        template <class A>
        inline bool any(batch_bool<float, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_i32x4_bitmask(self) != 0;
        }
        template <class A>
        inline bool any(batch_bool<double, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_i64x2_bitmask(self) != 0;
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline bool any(batch_bool<T, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_i8x16_bitmask(self) != 0;
        }

        // bitwise_and
        template <class A, class T>
        inline batch<T, A> bitwise_and(batch<T, A> const& self, batch<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_and(self, other);
        }

        template <class A, class T>
        inline batch_bool<T, A> bitwise_and(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_and(self, other);
        }

        // bitwise_andnot
        template <class A, class T>
        inline batch<T, A> bitwise_andnot(batch<T, A> const& self, batch<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_andnot(self, other);
        }

        template <class A, class T>
        inline batch_bool<T, A> bitwise_andnot(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_andnot(self, other);
        }

        // bitwise_or
        template <class A, class T>
        inline batch<T, A> bitwise_or(batch<T, A> const& self, batch<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(self, other);
        }

        template <class A, class T>
        inline batch_bool<T, A> bitwise_or(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(self, other);
        }

        // bitwise_lshift
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_lshift(batch<T, A> const& self, int32_t other, requires_arch<wasm>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
            {
                return wasm_v128_and(wasm_i8x16_splat(0xFF << other), wasm_i32x4_shl(self, other));
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
            {
                return wasm_i16x8_shl(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                return wasm_i32x4_shl(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
            {
                return wasm_i64x2_shl(self, other);
            }
            else
            {
                assert(false && "unsupported arch/op combination");
                return {};
            }
        }

        // bitwise_rshift
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_rshift(batch<T, A> const& self, int32_t other, requires_arch<wasm>) noexcept
        {
            if (std::is_signed<T>::value)
            {
                XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
                {
                    return wasm_i8x16_shr(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
                {
                    return wasm_i16x8_shr(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
                {
                    return wasm_i32x4_shr(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
                {
                    return wasm_i64x2_shr(self, other);
                }
                else
                {
                    assert(false && "unsupported arch/op combination");
                    return {};
                }
            }
            else
            {
                XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
                {
                    return wasm_v128_and(wasm_i8x16_splat(0xFF >> other), wasm_u32x4_shr(self, other));
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
                {
                    return wasm_u16x8_shr(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
                {
                    return wasm_u32x4_shr(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
                {
                    return wasm_u64x2_shr(self, other);
                }
                else
                {
                    assert(false && "unsupported arch/op combination");
                    return {};
                }
            }
        }

        // bitwise_not
        template <class A, class T>
        inline batch<T, A> bitwise_not(batch<T, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_v128_not(self);
        }

        template <class A, class T>
        inline batch_bool<T, A> bitwise_not(batch_bool<T, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_v128_not(self);
        }

        // bitwise_xor
        template <class A, class T>
        inline batch<T, A> bitwise_xor(batch<T, A> const& self, batch<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_xor(self, other);
        }

        template <class A, class T>
        inline batch_bool<T, A> bitwise_xor(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_xor(self, other);
        }

        // broadcast
        template <class A>
        batch<float, A> inline broadcast(float val, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_splat(val);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> broadcast(T val, requires_arch<wasm>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
            {
                return wasm_i8x16_splat(val);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
            {
                return wasm_i16x8_splat(val);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                return wasm_i32x4_splat(val);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
            {
                return wasm_i64x2_splat(val);
            }
            else
            {
                assert(false && "unsupported arch/op combination");
                return {};
            }
        }
        template <class A>
        inline batch<double, A> broadcast(double val, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_splat(val);
        }

        // ceil
        template <class A>
        inline batch<float, A> ceil(batch<float, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_ceil(self);
        }
        template <class A>
        inline batch<double, A> ceil(batch<double, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_ceil(self);
        }

        // div
        template <class A>
        inline batch<float, A> div(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_div(self, other);
        }
        template <class A>
        inline batch<double, A> div(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_div(self, other);
        }

        // eq
        template <class A>
        inline batch_bool<float, A> eq(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_eq(self, other);
        }
        template <class A>
        inline batch_bool<float, A> eq(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_eq(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<wasm>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
            {
                return wasm_i8x16_eq(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
            {
                return wasm_i16x8_eq(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                return wasm_i32x4_eq(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
            {
                return wasm_i64x2_eq(self, other);
            }
            else
            {
                assert(false && "unsupported arch/op combination");
                return {};
            }
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> eq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<wasm>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
            {
                return wasm_i8x16_eq(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
            {
                return wasm_i16x8_eq(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                return wasm_i32x4_eq(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
            {
                return wasm_i64x2_eq(self, other);
            }
            else
            {
                assert(false && "unsupported arch/op combination");
                return {};
            }
        }
        template <class A>
        inline batch_bool<double, A> eq(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_eq(self, other);
        }
        template <class A>
        inline batch_bool<double, A> eq(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_eq(self, other);
        }

        // floor
        template <class A>
        inline batch<float, A> floor(batch<float, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_floor(self);
        }

        template <class A>
        inline batch<double, A> floor(batch<double, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_floor(self);
        }

        // ge
        template <class A>
        inline batch_bool<float, A> ge(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_ge(self, other);
        }
        template <class A>
        inline batch_bool<double, A> ge(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_ge(self, other);
        }

        // gt
        template <class A>
        inline batch_bool<float, A> gt(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_gt(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<wasm>) noexcept
        {
            if (std::is_signed<T>::value)
            {
                XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
                {
                    return wasm_i8x16_gt(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
                {
                    return wasm_i16x8_gt(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
                {
                    return wasm_i32x4_gt(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
                {
                    return wasm_i64x2_gt(self, other);
                }
            }
            else
            {
                XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
                {
                    return wasm_u8x16_gt(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
                {
                    return wasm_u16x8_gt(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
                {
                    return wasm_u32x4_gt(self, other);
                }
                else
                {
                    return gt(self, other, generic {});
                }
            }
        }

        template <class A>
        inline batch_bool<double, A> gt(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_gt(self, other);
        }

        // insert
        template <class A, size_t I>
        inline batch<float, A> insert(batch<float, A> const& self, float val, index<I> pos, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_replace_lane(self, pos, val);
        }
        template <class A, class T, size_t I, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> insert(batch<T, A> const& self, T val, index<I> pos, requires_arch<wasm>) noexcept
        {
            if (std::is_signed<T>::value)
            {
                XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
                {
                    return wasm_i8x16_replace_lane(self, pos, val);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
                {
                    return wasm_i16x8_replace_lane(self, pos, val);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
                {
                    return wasm_i32x4_replace_lane(self, pos, val);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
                {
                    return wasm_i64x2_replace_lane(self, pos, val);
                }
                else
                {
                    assert(false && "unsupported arch/op combination");
                    return {};
                }
            }
            else
            {
                XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
                {
                    return wasm_u8x16_replace_lane(self, pos, val);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
                {
                    return wasm_u16x8_replace_lane(self, pos, val);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
                {
                    return wasm_u32x4_replace_lane(self, pos, val);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
                {
                    return wasm_u64x2_replace_lane(self, pos, val);
                }
                else
                {
                    assert(false && "unsupported arch/op combination");
                    return {};
                }
            }
        }

        template <class A, size_t I>
        inline batch<double, A> insert(batch<double, A> const& self, double val, index<I> pos, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_replace_lane(self, pos, val);
        }

        // isnan
        template <class A>
        inline batch_bool<float, A> isnan(batch<float, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(wasm_f32x4_ne(self, self), wasm_f32x4_ne(self, self));
        }
        template <class A>
        inline batch_bool<double, A> isnan(batch<double, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(wasm_f64x2_ne(self, self), wasm_f64x2_ne(self, self));
        }

        // le
        template <class A>
        inline batch_bool<float, A> le(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_le(self, other);
        }
        template <class A>
        inline batch_bool<double, A> le(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_le(self, other);
        }

        // load_aligned
        template <class A>
        inline batch<float, A> load_aligned(float const* mem, convert<float>, requires_arch<wasm>) noexcept
        {
            // Assuming that mem is aligned properly, you can use wasm_v128_load to load the mem.
            return wasm_v128_load(mem);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> load_aligned(T const* mem, convert<T>, requires_arch<wasm>) noexcept
        {
            // Assuming that mem is aligned properly, you can use wasm_v128_load to load the mem.
            return wasm_v128_load((v128_t const*)mem);
        }
        template <class A>
        inline batch<double, A> load_aligned(double const* mem, convert<double>, requires_arch<wasm>) noexcept
        {
            // Assuming that mem is aligned properly, you can use wasm_v128_load to load the mem.
            return wasm_v128_load(mem);
        }

        // load_unaligned
        template <class A>
        inline batch<float, A> load_unaligned(float const* mem, convert<float>, requires_arch<wasm>) noexcept
        {
            return wasm_v128_load(mem);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> load_unaligned(T const* mem, convert<T>, requires_arch<wasm>) noexcept
        {
            return wasm_v128_load((v128_t const*)mem);
        }
        template <class A>
        inline batch<double, A> load_unaligned(double const* mem, convert<double>, requires_arch<wasm>) noexcept
        {
            return wasm_v128_load(mem);
        }

        // lt
        template <class A>
        inline batch_bool<float, A> lt(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_lt(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> lt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<wasm>) noexcept
        {
            if (std::is_signed<T>::value)
            {
                XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
                {
                    return wasm_i8x16_lt(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
                {
                    return wasm_i16x8_lt(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
                {
                    return wasm_i32x4_lt(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
                {
                    return wasm_i64x2_lt(self, other);
                }
                else
                {
                    assert(false && "unsupported arch/op combination");
                    return {};
                }
            }
            else
            {
                XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
                {
                    return wasm_u8x16_lt(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
                {
                    return wasm_u16x8_lt(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
                {
                    return wasm_u32x4_lt(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
                {
                    return lt(self, other, generic {});
                }
                else
                {
                    assert(false && "unsupported arch/op combination");
                    return {};
                }
            }
        }

        template <class A>
        inline batch_bool<double, A> lt(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_lt(self, other);
        }

        // mask
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline uint64_t mask(batch_bool<T, A> const& self, requires_arch<wasm>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
            {
                return wasm_i8x16_bitmask(self);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
            {
                return wasm_i16x8_bitmask(self);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                return wasm_i32x4_bitmask(self);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
            {
                return wasm_i64x2_bitmask(self);
            }
            else
            {
                assert(false && "unsupported arch/op combination");
                return {};
            }
        }
        template <class A>
        inline uint64_t mask(batch_bool<float, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_i32x4_bitmask(self);
        }

        template <class A>
        inline uint64_t mask(batch_bool<double, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_i64x2_bitmask(self);
        }

        // max
        template <class A>
        inline batch<float, A> max(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_pmax(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return select(self > other, self, other);
        }
        template <class A>
        inline batch<double, A> max(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_pmax(self, other);
        }

        // min
        template <class A>
        inline batch<float, A> min(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_pmin(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return select(self <= other, self, other);
        }
        template <class A>
        inline batch<double, A> min(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_pmin(self, other);
        }

        // mul
        template <class A>
        inline batch<float, A> mul(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_mul(self, other);
        }
        template <class A>
        inline batch<double, A> mul(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_mul(self, other);
        }

        // neg
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> neg(batch<T, A> const& self, requires_arch<wasm>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
            {
                return wasm_i8x16_neg(self);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
            {
                return wasm_i16x8_neg(self);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                return wasm_i32x4_neg(self);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
            {
                return wasm_i64x2_neg(self);
            }
            else
            {
                assert(false && "unsupported arch/op combination");
                return {};
            }
        }

        template <class A>
        inline batch<float, A> neg(batch<float, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_neg(self);
        }

        template <class A>
        inline batch<double, A> neg(batch<double, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_neg(self);
        }

        // neq
        template <class A>
        inline batch_bool<float, A> neq(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_ne(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return ~(self == other);
        }
        template <class A>
        inline batch_bool<float, A> neq(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_ne(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> neq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return ~(self == other);
        }

        template <class A>
        inline batch_bool<double, A> neq(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_ne(self, other);
        }
        template <class A>
        inline batch_bool<double, A> neq(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_ne(self, other);
        }

        // reciprocal
        template <class A>
        inline batch<float, A> reciprocal(batch<float, A> const& self, requires_arch<wasm>) noexcept
        {
            v128_t one = wasm_f32x4_splat(1.0f);
            return wasm_f32x4_div(one, self);
        }
        template <class A>
        inline batch<double, A> reciprocal(batch<double, A> const& self, requires_arch<wasm>) noexcept
        {
            v128_t one = wasm_f64x2_splat(1.0);
            return wasm_f64x2_div(one, self);
        }

        // rsqrt
        template <class A>
        inline batch<float, A> rsqrt(batch<float, A> const& self, requires_arch<wasm>) noexcept
        {
            v128_t one = wasm_f32x4_splat(1.0f);
            return wasm_f32x4_div(one, wasm_f32x4_sqrt(self));
        }
        template <class A>
        inline batch<double, A> rsqrt(batch<double, A> const& self, requires_arch<wasm>) noexcept
        {
            v128_t one = wasm_f64x2_splat(1.0);
            return wasm_f64x2_div(one, wasm_f64x2_sqrt(self));
        }

        // select
        template <class A>
        inline batch<float, A> select(batch_bool<float, A> const& cond, batch<float, A> const& true_br, batch<float, A> const& false_br, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(wasm_v128_and(cond, true_br), wasm_v128_andnot(false_br, cond));
        }

        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(wasm_v128_and(cond, true_br), wasm_v128_andnot(false_br, cond));
        }
        template <class A, class T, bool... Values, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> select(batch_bool_constant<batch<T, A>, Values...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<wasm>) noexcept
        {
            return select(batch_bool<T, A> { Values... }, true_br, false_br, wasm {});
        }
        template <class A>
        inline batch<double, A> select(batch_bool<double, A> const& cond, batch<double, A> const& true_br, batch<double, A> const& false_br, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(wasm_v128_and(cond, true_br), wasm_v128_andnot(false_br, cond));
        }

        // set
        template <class A, class... Values>
        inline batch<float, A> set(batch<float, A> const&, requires_arch<wasm>, Values... values) noexcept
        {
            static_assert(sizeof...(Values) == batch<float, A>::size, "consistent init");
            return wasm_f32x4_make(values...);
        }

        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> set(batch<T, A> const&, requires_arch<wasm>, T v0, T v1) noexcept
        {
            return wasm_i64x2_make(v0, v1);
        }

        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> set(batch<T, A> const&, requires_arch<wasm>, T v0, T v1, T v2, T v3) noexcept
        {
            return wasm_i32x4_make(v0, v1, v2, v3);
        }

        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> set(batch<T, A> const&, requires_arch<wasm>, T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) noexcept
        {
            return wasm_i16x8_make(v0, v1, v2, v3, v4, v5, v6, v7);
        }

        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> set(batch<T, A> const&, requires_arch<wasm>, T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) noexcept
        {
            return wasm_i8x16_make(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
        }

        template <class A, class... Values>
        inline batch<double, A> set(batch<double, A> const&, requires_arch<wasm>, Values... values) noexcept
        {
            static_assert(sizeof...(Values) == batch<double, A>::size, "consistent init");
            return wasm_f64x2_make(values...);
        }

        template <class A, class T, class... Values, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> set(batch_bool<T, A> const&, requires_arch<wasm>, Values... values) noexcept
        {
            return set(batch<T, A>(), A {}, static_cast<T>(values ? -1LL : 0LL)...).data;
        }

        template <class A, class... Values>
        inline batch_bool<float, A> set(batch_bool<float, A> const&, requires_arch<wasm>, Values... values) noexcept
        {
            static_assert(sizeof...(Values) == batch_bool<float, A>::size, "consistent init");
            return set(batch<int32_t, A>(), A {}, static_cast<int32_t>(values ? -1LL : 0LL)...).data;
        }

        template <class A, class... Values>
        inline batch_bool<double, A> set(batch_bool<double, A> const&, requires_arch<wasm>, Values... values) noexcept
        {
            static_assert(sizeof...(Values) == batch_bool<double, A>::size, "consistent init");
            return set(batch<int64_t, A>(), A {}, static_cast<int64_t>(values ? -1LL : 0LL)...).data;
        }

        // store_aligned
        template <class A>
        inline void store_aligned(float* mem, batch<float, A> const& self, requires_arch<wasm>) noexcept
        {
            // Assuming that mem is aligned properly, you can use wasm_v128_store to store the batch.
            return wasm_v128_store(mem, self);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline void store_aligned(T* mem, batch<T, A> const& self, requires_arch<wasm>) noexcept
        {
            // Assuming that mem is aligned properly, you can use wasm_v128_store to store the batch.
            return wasm_v128_store((v128_t*)mem, self);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline void store_aligned(T* mem, batch_bool<T, A> const& self, requires_arch<wasm>) noexcept
        {
            // Assuming that mem is aligned properly, you can use wasm_v128_store to store the batch.
            return wasm_v128_store((v128_t*)mem, self);
        }
        template <class A>
        inline void store_aligned(double* mem, batch<double, A> const& self, requires_arch<wasm>) noexcept
        {
            // Assuming that mem is aligned properly, you can use wasm_v128_store to store the batch.
            return wasm_v128_store(mem, self);
        }

        // store_unaligned
        template <class A>
        inline void store_unaligned(float* mem, batch<float, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_v128_store(mem, self);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline void store_unaligned(T* mem, batch<T, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_v128_store((v128_t*)mem, self);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline void store_unaligned(T* mem, batch_bool<T, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_v128_store((v128_t*)mem, self);
        }
        template <class A>
        inline void store_unaligned(double* mem, batch<double, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_v128_store(mem, self);
        }

        // sub
        template <class A>
        inline batch<float, A> sub(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_sub(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> sub(batch<T, A> const& self, batch<T, A> const& other, requires_arch<wasm>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
            {
                return wasm_i8x16_sub(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
            {
                return wasm_i16x8_sub(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                return wasm_i32x4_sub(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
            {
                return wasm_i64x2_sub(self, other);
            }
            else
            {
                assert(false && "unsupported arch/op combination");
                return {};
            }
        }
        template <class A>
        inline batch<double, A> sub(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_sub(self, other);
        }

        // sqrt
        template <class A>
        inline batch<float, A> sqrt(batch<float, A> const& val, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_sqrt(val);
        }
        template <class A>
        inline batch<double, A> sqrt(batch<double, A> const& val, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_sqrt(val);
        }

        // trunc
        template <class A>
        inline batch<float, A> trunc(batch<float, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_trunc(self);
        }
        template <class A>
        inline batch<double, A> trunc(batch<double, A> const& self, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_trunc(self);
        }
    }
}

#endif