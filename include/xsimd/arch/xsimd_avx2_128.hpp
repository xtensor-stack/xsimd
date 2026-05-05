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

#include <type_traits>

#include "../types/xsimd_avx2_register.hpp"
#include "../types/xsimd_batch_constant.hpp"

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

        namespace detail
        {
            XSIMD_INLINE __m128i maskload_128(int32_t const* mem, __m128i mask) noexcept
            {
                return _mm_maskload_epi32(mem, mask);
            }
            XSIMD_INLINE __m128i maskload_128(long long const* mem, __m128i mask) noexcept
            {
                return _mm_maskload_epi64(mem, mask);
            }
            XSIMD_INLINE void maskstore_128(int32_t* mem, __m128i mask, __m128i src) noexcept
            {
                _mm_maskstore_epi32(mem, mask, src);
            }
            XSIMD_INLINE void maskstore_128(long long* mem, __m128i mask, __m128i src) noexcept
            {
                _mm_maskstore_epi64(mem, mask, src);
            }
        }

        // load_masked / store_masked. AVX2-128 has _mm_maskload/store for
        // 32/64-bit integers only; 8/16-bit fall back to the common scalar
        // path. Aligned and unaligned go through the same intrinsic — masked-
        // off lanes do not fault regardless of alignment.
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<int32_t, A> load_masked(int32_t const* mem, batch_bool_constant<int32_t, A, Values...> mask, convert<int32_t>, Mode, requires_arch<avx2_128>) noexcept
        {
            return detail::maskload_128(mem, mask.as_batch());
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<uint32_t, A> load_masked(uint32_t const* mem, batch_bool_constant<uint32_t, A, Values...> mask, convert<uint32_t>, Mode, requires_arch<avx2_128>) noexcept
        {
            return detail::maskload_128(reinterpret_cast<int32_t const*>(mem), mask.as_batch());
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<int64_t, A> load_masked(int64_t const* mem, batch_bool_constant<int64_t, A, Values...> mask, convert<int64_t>, Mode, requires_arch<avx2_128>) noexcept
        {
            return detail::maskload_128(reinterpret_cast<long long const*>(mem), mask.as_batch());
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE batch<uint64_t, A> load_masked(uint64_t const* mem, batch_bool_constant<uint64_t, A, Values...> mask, convert<uint64_t>, Mode, requires_arch<avx2_128>) noexcept
        {
            return detail::maskload_128(reinterpret_cast<long long const*>(mem), mask.as_batch());
        }

        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(int32_t* mem, batch<int32_t, A> const& src, batch_bool_constant<int32_t, A, Values...> mask, Mode, requires_arch<avx2_128>) noexcept
        {
            detail::maskstore_128(mem, mask.as_batch(), src);
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(uint32_t* mem, batch<uint32_t, A> const& src, batch_bool_constant<uint32_t, A, Values...> mask, Mode, requires_arch<avx2_128>) noexcept
        {
            detail::maskstore_128(reinterpret_cast<int32_t*>(mem), mask.as_batch(), src);
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(int64_t* mem, batch<int64_t, A> const& src, batch_bool_constant<int64_t, A, Values...> mask, Mode, requires_arch<avx2_128>) noexcept
        {
            detail::maskstore_128(reinterpret_cast<long long*>(mem), mask.as_batch(), src);
        }
        template <class A, bool... Values, class Mode>
        XSIMD_INLINE void store_masked(uint64_t* mem, batch<uint64_t, A> const& src, batch_bool_constant<uint64_t, A, Values...> mask, Mode, requires_arch<avx2_128>) noexcept
        {
            detail::maskstore_128(reinterpret_cast<long long*>(mem), mask.as_batch(), src);
        }

        // Runtime-mask path: route through the same helpers as the constant-
        // mask overloads. Templated on T so the dispatcher does not need to
        // see four near-identical functions; the runtime-mask overloads of
        // the common arch take ``batch_bool<T, A>``, not the four signed/
        // unsigned int variants, so there is no ambiguity here.
        template <class A, class T, class Mode>
        XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8), batch<T, A>>
        load_masked(T const* mem, batch_bool<T, A> mask, convert<T>, Mode, requires_arch<avx2_128>) noexcept
        {
            using int_t = std::conditional_t<sizeof(T) == 4, int32_t, long long>;
            return detail::maskload_128(reinterpret_cast<int_t const*>(mem), __m128i(mask));
        }

        template <class A, class T, class Mode>
        XSIMD_INLINE std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8), void>
        store_masked(T* mem, batch<T, A> const& src, batch_bool<T, A> mask, Mode, requires_arch<avx2_128>) noexcept
        {
            using int_t = std::conditional_t<sizeof(T) == 4, int32_t, long long>;
            detail::maskstore_128(reinterpret_cast<int_t*>(mem), __m128i(mask), __m128i(src));
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
