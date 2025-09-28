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

#ifndef XSIMD_SSE3_HPP
#define XSIMD_SSE3_HPP

#include "../types/xsimd_sse3_register.hpp"
#include <type_traits>

namespace xsimd
{

    namespace kernel
    {
        using namespace types;

        // haddp
        template <class A>
        XSIMD_INLINE batch<float, A> haddp(batch<float, A> const* row, requires_arch<sse3>) noexcept
        {
            return _mm_hadd_ps(_mm_hadd_ps(row[0], row[1]),
                               _mm_hadd_ps(row[2], row[3]));
        }
        template <class A>
        XSIMD_INLINE batch<double, A> haddp(batch<double, A> const* row, requires_arch<sse3>) noexcept
        {
            return _mm_hadd_pd(row[0], row[1]);
        }

        // load_unaligned
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> load_unaligned(T const* mem, convert<T>, requires_arch<sse3>) noexcept
        {
            return _mm_lddqu_si128((__m128i const*)mem);
        }

        // reduce_add
        template <class A>
        XSIMD_INLINE float reduce_add(batch<float, A> const& self, requires_arch<sse3>) noexcept
        {
            __m128 tmp0 = _mm_hadd_ps(self, self);
            __m128 tmp1 = _mm_hadd_ps(tmp0, tmp0);
            return _mm_cvtss_f32(tmp1);
        }

        // reduce_mul
        template <class A>
        XSIMD_INLINE float reduce_mul(batch<float, A> const& self, requires_arch<sse3>) noexcept
        {
            __m128 tmp1 = _mm_mul_ps(self, _mm_movehl_ps(self, self));
            __m128 tmp2 = _mm_mul_ps(tmp1, _mm_movehdup_ps(tmp1));
            return _mm_cvtss_f32(tmp2);
        }

        // store<batch_bool>
        namespace detail {
            template <class T>
            XSIMD_INLINE void store_bool_sse3(__m128i b, bool* mem, T) noexcept {
                // GCC <12 have missing or buggy unaligned store intrinsics; use memcpy to work around this.
                // GCC/Clang/MSVC will turn it into the correct store.
                XSIMD_IF_CONSTEXPR(sizeof(T) == 1) {
                    // negate mask to convert to 0 or 1
                    auto val = _mm_sub_epi8(_mm_set1_epi8(0), b);
                    memcpy(mem, &val, sizeof(val));
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 2) {
                    auto packed = _mm_packs_epi16(b, b);
                    uint64_t val = _mm_extract_epi64(_mm_sub_epi8(_mm_set1_epi8(0), packed), 0);
                    memcpy(mem, &val, sizeof(val));
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 4) {
                    const auto bmask = _mm_set_epi8(
                        -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, 12,  8,  4,  0);
                    auto packed = _mm_shuffle_epi8(b, bmask);
                    uint32_t val = _mm_extract_epi32(_mm_sub_epi8(_mm_set1_epi8(0), packed), 0);
                    memcpy(mem, &val, sizeof(val));
                }
                else XSIMD_IF_CONSTEXPR(sizeof(T) == 8) {
                    const auto bmask = _mm_set_epi8(
                        -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1, -1,  8,  0);
                    auto packed = _mm_shuffle_epi8(b, bmask);
                    uint16_t val = _mm_extract_epi16(_mm_sub_epi8(_mm_set1_epi8(0), packed), 0);
                    memcpy(mem, &val, sizeof(val));
                }
            }

            XSIMD_INLINE __m128i sse_to_i(__m128 x) { return _mm_castps_si128(x); }
            XSIMD_INLINE __m128i sse_to_i(__m128d x) { return _mm_castpd_si128(x); }
            XSIMD_INLINE __m128i sse_to_i(__m128i x) { return x; }
        }

        template <class T, class A>
        XSIMD_INLINE void store(batch_bool<T, A> b, bool* mem, requires_arch<sse3>) noexcept
        {
            detail::store_bool_sse3(detail::sse_to_i(b), mem, T{});
        }

    }

}

#endif
