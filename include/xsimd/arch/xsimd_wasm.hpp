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
        inline batch<float, A> abs(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f32x4_abs(self, other);
        }

        template <class A>
        inline batch<double, A> abs(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_f64x2_abs(self, other);
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

        // bitwise_xor
        template <class A>
        inline batch<float, A> bitwise_xor(batch<float, A> const& self, batch<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_xor(self, other);
        }
        template <class A>
        inline batch_bool<float, A> bitwise_xor(batch_bool<float, A> const& self, batch_bool<float, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_xor(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_xor(batch<T, A> const& self, batch<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(self, other);
        }
        template <class A>
        inline batch<double, A> bitwise_xor(batch<double, A> const& self, batch<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(self, other);
        }
        template <class A>
        inline batch_bool<double, A> bitwise_xor(batch_bool<double, A> const& self, batch_bool<double, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_xor(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(self, other);
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

        // select
        template <class A>
        inline batch<float, A> select(batch_bool<float, A> const& cond, batch<float, A> const& true_br, batch<float, A> const& false_br, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(wasm_v128_and(cond, true_br), wasm_v128_andnot(cond, false_br));
        }

        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(wasm_v128_and(cond, true_br), wasm_v128_andnot(cond, false_br));
        }
        template <class A, class T, bool... Values, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> select(batch_bool_constant<batch<T, A>, Values...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<wasm>) noexcept
        {
            return select(batch_bool<T, A> { Values... }, true_br, false_br, wasm {});
        }
        template <class A>
        inline batch<double, A> select(batch_bool<double, A> const& cond, batch<double, A> const& true_br, batch<double, A> const& false_br, requires_arch<wasm>) noexcept
        {
            return wasm_v128_or(wasm_v128_and(cond, true_br), wasm_v128_andnot(cond, false_br));
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
        inline batch<T, A> set(batch<T, A> const&, requires_arch<wasm>, T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7,
                            T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) noexcept
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
    }
}

#endif