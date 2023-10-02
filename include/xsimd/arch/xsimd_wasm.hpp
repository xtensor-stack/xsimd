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
        template <class A, class T>
        inline batch<T, A> abs(batch<T, A> const& self, requires_arch<wasm>) noexcept
        {
            XSIMD_IF_CONSTEXPR(std::is_integral_v<T> && std::is_signed_v<T>)
            {
                XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
                {
                    return wasm_i8x16_abs(self);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
                {
                    return  wasm_i16x8_abs(self);
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
            else XSIMD_IF_CONSTEXPR(std::is_floating_point_v<T>)
            {
                XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
                {
                    return wasm_f32x4_abs(self);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
                {
                    return wasm_f64x2_abs(self);
                }
                else
                {
                    assert(false && "unsupported arch/op combination");
                    return {};
                }
            }
            else
            {
                assert(false && "unsupported arch/op combination");
                return {};
            }
        }

        //set
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

        //store_aligned
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline void store_aligned(T* mem, batch<T, A> const& self, requires_arch<wasm>) noexcept
        {
            // Assuming that mem is aligned properly, you can use wasm_v128_store to store the batch.
            return wasm_v128_store((v128_t*)mem, self);
        }
    }
}

#endif