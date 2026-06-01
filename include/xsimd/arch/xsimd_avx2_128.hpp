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

#ifndef XSIMD_AVX2_128_HPP
#define XSIMD_AVX2_128_HPP

#include "../types/xsimd_avx2_register.hpp"
#include "../types/xsimd_batch_constant.hpp"

#include <type_traits>

namespace xsimd
{
    namespace kernel
    {
        using namespace types;

        // select
        template <class A, class T, bool... Values, class = std::enable_if_t<std::is_integral<T>::value>>
        XSIMD_INLINE batch<T, A> select(batch_bool_constant<T, A, Values...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<avx2_128>) noexcept
        {
            constexpr int mask = batch_bool_constant<T, A, Values...>::mask();
            XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                return _mm_blend_epi32(false_br, true_br, mask);
            }
            else
            {
                return select(batch_bool_constant<T, A, Values...>(), true_br, false_br, avx_128 {});
            }
        }

        // bitwise_lshift
        template <class A, class T, class = std::enable_if_t<std::is_integral<T>::value>>
        XSIMD_INLINE batch<T, A> bitwise_lshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2_128>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                return _mm_sllv_epi32(self, other);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
            {
                return _mm_sllv_epi64(self, other);
            }
            else
            {
                return bitwise_lshift(self, other, avx {});
            }
        }

        // bitwise_rshift
        template <class A, class T, class = std::enable_if_t<std::is_integral<T>::value>>
        XSIMD_INLINE batch<T, A> bitwise_rshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2_128>) noexcept
        {
            if (std::is_signed<T>::value)
            {
                XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
                {
                    return _mm_srav_epi32(self, other);
                }
                else
                {
                    return bitwise_rshift(self, other, avx_128 {});
                }
            }
            else
            {
                XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
                {
                    return _mm_srlv_epi32(self, other);
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
                {
                    return _mm_srlv_epi64(self, other);
                }
                else
                {
                    return bitwise_rshift(self, other, avx_128 {});
                }
            }
        }

        // load_masked — native 128-bit integer masked loads. Tagged on avx2_128
        // because the vpmaskmov* intrinsics require AVX2; an AVX1-only build routes
        // integer masked memory through the float path in xsimd_common_memory.hpp.
        // Any arch with a native masked path provides its own exact-tag overload that
        // out-ranks this one, so no cross-arch exclusion is needed here.
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<int32_t, A> load_masked(int32_t const* mem, batch_bool_constant<int32_t, A, Values...> mask, convert<int32_t>, Mode, requires_arch<avx2_128>) noexcept
        {
            return _mm_maskload_epi32(mem, mask.as_batch());
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<uint32_t, A> load_masked(uint32_t const* mem, batch_bool_constant<uint32_t, A, Values...> mask, convert<uint32_t>, Mode, requires_arch<avx2_128>) noexcept
        {
            return _mm_maskload_epi32(reinterpret_cast<int32_t const*>(mem), mask.as_batch());
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<int64_t, A> load_masked(int64_t const* mem, batch_bool_constant<int64_t, A, Values...> mask, convert<int64_t>, Mode, requires_arch<avx2_128>) noexcept
        {
            return _mm_maskload_epi64(reinterpret_cast<long long const*>(mem), mask.as_batch());
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<uint64_t, A> load_masked(uint64_t const* mem, batch_bool_constant<uint64_t, A, Values...> mask, convert<uint64_t>, Mode, requires_arch<avx2_128>) noexcept
        {
            return _mm_maskload_epi64(reinterpret_cast<long long const*>(mem), mask.as_batch());
        }

        // store_masked — native 128-bit integer masked stores (see load note above).
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(int32_t* mem, batch<int32_t, A> const& src, batch_bool_constant<int32_t, A, Values...> mask, Mode, requires_arch<avx2_128>) noexcept
        {
            return _mm_maskstore_epi32(mem, mask.as_batch(), src);
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(uint32_t* mem, batch<uint32_t, A> const& src, batch_bool_constant<uint32_t, A, Values...> mask, Mode, requires_arch<avx2_128>) noexcept
        {
            return _mm_maskstore_epi32(reinterpret_cast<int32_t*>(mem), mask.as_batch(), src);
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(int64_t* mem, batch<int64_t, A> const& src, batch_bool_constant<int64_t, A, Values...> mask, Mode, requires_arch<avx2_128>) noexcept
        {
            return _mm_maskstore_epi64(reinterpret_cast<long long*>(mem), mask.as_batch(), src);
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(uint64_t* mem, batch<uint64_t, A> const& src, batch_bool_constant<uint64_t, A, Values...> mask, Mode, requires_arch<avx2_128>) noexcept
        {
            return _mm_maskstore_epi64(reinterpret_cast<long long*>(mem), mask.as_batch(), src);
        }

        // gather
        template <class T, class A, class U, detail::enable_sized_integral_t<T, 4> = 0, detail::enable_sized_integral_t<U, 4> = 0>
        XSIMD_INLINE batch<T, A> gather(batch<T, A> const&, T const* src, batch<U, A> const& index,
                                        kernel::requires_arch<avx2_128>) noexcept
        {
            return _mm_i32gather_epi32(reinterpret_cast<const int*>(src), index, sizeof(T));
        }

        template <class T, class A, class U, detail::enable_sized_integral_t<T, 8> = 0, detail::enable_sized_integral_t<U, 8> = 0>
        XSIMD_INLINE batch<T, A> gather(batch<T, A> const&, T const* src, batch<U, A> const& index,
                                        kernel::requires_arch<avx2_128>) noexcept
        {
            return _mm_i64gather_epi64(reinterpret_cast<const long long int*>(src), index, sizeof(T));
        }

        template <class A, class U,
                  detail::enable_sized_integral_t<U, 4> = 0>
        XSIMD_INLINE batch<float, A> gather(batch<float, A> const&, float const* src,
                                            batch<U, A> const& index,
                                            kernel::requires_arch<avx2_128>) noexcept
        {
            return _mm_i32gather_ps(src, index, sizeof(float));
        }

        template <class A, class U, detail::enable_sized_integral_t<U, 8> = 0>
        XSIMD_INLINE batch<double, A> gather(batch<double, A> const&, double const* src,
                                             batch<U, A> const& index,
                                             requires_arch<avx2_128>) noexcept
        {
            return _mm_i64gather_pd(src, index, sizeof(double));
        }
    }
}

#endif
