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

#ifndef XSIMD_ALTIVEC_HPP
#define XSIMD_ALTIVEC_HPP

#include <complex>
#include <limits>
#include <type_traits>

#include "../types/xsimd_altivec_register.hpp"

namespace xsimd
{
    template <typename T, class A, bool... Values>
    struct batch_bool_constant;

    template <class T_out, class T_in, class A>
    XSIMD_INLINE batch<T_out, A> bitwise_cast(batch<T_in, A> const& x) noexcept;

    template <typename T, class A, T... Values>
    struct batch_constant;

    namespace kernel
    {
#if 0
        using namespace types;

        namespace detail
        {
            constexpr uint32_t shuffle(uint32_t w, uint32_t x, uint32_t y, uint32_t z)
            {
                return (z << 6) | (y << 4) | (x << 2) | w;
            }
            constexpr uint32_t shuffle(uint32_t x, uint32_t y)
            {
                return (y << 1) | x;
            }

            constexpr uint32_t mod_shuffle(uint32_t w, uint32_t x, uint32_t y, uint32_t z)
            {
                return shuffle(w % 4, x % 4, y % 4, z % 4);
            }

            constexpr uint32_t mod_shuffle(uint32_t w, uint32_t x)
            {
                return shuffle(w % 2, x % 2);
            }
        }

        // fwd
        template <class A, class T, size_t I>
        XSIMD_INLINE batch<T, A> insert(batch<T, A> const& self, T val, index<I>, requires_arch<common>) noexcept;
        template <class A, typename T, typename ITy, ITy... Indices>
        XSIMD_INLINE batch<T, A> shuffle(batch<T, A> const& x, batch<T, A> const& y, batch_constant<ITy, A, Indices...>, requires_arch<common>) noexcept;
        template <class A, class T>
        XSIMD_INLINE batch<T, A> avg(batch<T, A> const&, batch<T, A> const&, requires_arch<common>) noexcept;
        template <class A, class T>
        XSIMD_INLINE batch<T, A> avgr(batch<T, A> const&, batch<T, A> const&, requires_arch<common>) noexcept;
#endif

        // abs
        template <class A>
        XSIMD_INLINE batch<float, A> abs(batch<float, A> const& self, requires_arch<altivec>) noexcept
        {
            return vec_abs(self);
        }

        // add
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> add(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_add(self, other);
        }

        // all
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE bool all(batch_bool<T, A> const& self, requires_arch<altivec>) noexcept
        {
            return vec_all_ne(self, vec_xor(self, self));
        }

        // any
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE bool any(batch_bool<T, A> const& self, requires_arch<altivec>) noexcept
        {
            return vec_any_ne(self, vec_xor(self, self));
        }

        // avgr
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> avgr(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_avg(self, other);
        }

        // avg
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> avg(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            constexpr auto nbit = 8 * sizeof(T) - 1;
            constexpr auto adj = ((self ^ other) << nbit) >> nbit;
            return avgr(self, other, A {}) - adj;
        }

        // batch_bool_cast
        template <class A, class T_out, class T_in>
        XSIMD_INLINE batch_bool<T_out, A> batch_bool_cast(batch_bool<T_in, A> const& self, batch_bool<T_out, A> const&, requires_arch<altivec>) noexcept
        {
            return { bitwise_cast<T_out>(batch<T_in, A>(self.data)).data };
        }

        // bitwise_and
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> bitwise_and(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_and(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch_bool<T, A> bitwise_and(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_and(self, other);
        }

        // bitwise_andnot
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> bitwise_andnot(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_nand(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch_bool<T, A> bitwise_andnot(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_nand(self, other);
        }

        // bitwise_lshift
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> bitwise_lshift(batch<T, A> const& self, int32_t other, requires_arch<altivec>) noexcept
        {
            using shift_type = as_unsigned_integer_t<T>;
            batch<shift_type, A> shift(static_cast<shift_type>(other));
            return vec_sl(self, shift);
        }

        // bitwise_not
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> bitwise_not(batch<T, A> const& self, requires_arch<altivec>) noexcept
        {
            return vec_nor(self, self);
        }
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch_bool<T, A> bitwise_not(batch_bool<T, A> const& self, requires_arch<altivec>) noexcept
        {
            return vec_nor(self, self);
        }

        // bitwise_or
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> bitwise_or(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_or(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch_bool<T, A> bitwise_or(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_or(self, other);
        }

        // bitwise_rshift
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> bitwise_rshift(batch<T, A> const& self, int32_t other, requires_arch<altivec>) noexcept
        {
            using shift_type = as_unsigned_integer_t<T>;
            batch<shift_type, A> shift(static_cast<shift_type>(other));
            return vec_sr(self, shift);
        }

        // bitwise_xor
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> bitwise_xor(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_xor(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> bitwise_xor(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_xor(self, other);
        }

        // bitwise_cast
        template <class A, class T_in, class T_out>
        XSIMD_INLINE batch<T_out, A> bitwise_cast(batch<T_in, A> const& self, batch<T_out, A> const&, requires_arch<altivec>) noexcept
        {
            return *reinterpret_cast<typename batch<T_out, A>::register_type const*>(&self.data);
        }

        // broadcast
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> broadcast(T val, requires_arch<altivec>) noexcept
        {
            return vec_splats(val);
        }
#if 0

        // store_complex
        namespace detail
        {
            // Override these methods in SSE-based archs, no need to override store_aligned / store_unaligned
            // complex_low
            template <class A>
            XSIMD_INLINE batch<float, A> complex_low(batch<std::complex<float>, A> const& self, requires_arch<altivec>) noexcept
            {
                return _mm_unpacklo_ps(self.real(), self.imag());
            }
            // complex_high
            template <class A>
            XSIMD_INLINE batch<float, A> complex_high(batch<std::complex<float>, A> const& self, requires_arch<altivec>) noexcept
            {
                return _mm_unpackhi_ps(self.real(), self.imag());
            }
            template <class A>
            XSIMD_INLINE batch<double, A> complex_low(batch<std::complex<double>, A> const& self, requires_arch<altivec>) noexcept
            {
                return _mm_unpacklo_pd(self.real(), self.imag());
            }
            template <class A>
            XSIMD_INLINE batch<double, A> complex_high(batch<std::complex<double>, A> const& self, requires_arch<altivec>) noexcept
            {
                return _mm_unpackhi_pd(self.real(), self.imag());
            }
        }

        // decr_if
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> decr_if(batch<T, A> const& self, batch_bool<T, A> const& mask, requires_arch<altivec>) noexcept
        {
            return self + batch<T, A>(mask.data);
        }

        // div
        template <class A>
        XSIMD_INLINE batch<float, A> div(batch<float, A> const& self, batch<float, A> const& other, requires_arch<altivec>) noexcept
        {
            return _mm_div_ps(self, other);
        }
        template <class A>
        XSIMD_INLINE batch<double, A> div(batch<double, A> const& self, batch<double, A> const& other, requires_arch<altivec>) noexcept
        {
            return _mm_div_pd(self, other);
        }

        // fast_cast
        namespace detail
        {
            template <class A>
            XSIMD_INLINE batch<float, A> fast_cast(batch<int32_t, A> const& self, batch<float, A> const&, requires_arch<altivec>) noexcept
            {
                return _mm_cvtepi32_ps(self);
            }

            template <class A>
            XSIMD_INLINE batch<double, A> fast_cast(batch<uint64_t, A> const& x, batch<double, A> const&, requires_arch<altivec>) noexcept
            {
                // from https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
                // adapted to altivec
                __m128i xH = _mm_srli_epi64(x, 32);
                xH = _mm_or_si128(xH, _mm_castpd_si128(_mm_set1_pd(19342813113834066795298816.))); //  2^84
                __m128i mask = _mm_setr_epi16(0xFFFF, 0xFFFF, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0x0000, 0x0000);
                __m128i xL = _mm_or_si128(_mm_and_si128(mask, x), _mm_andnot_si128(mask, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)))); //  2^52
                __m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(19342813118337666422669312.)); //  2^84 + 2^52
                return _mm_add_pd(f, _mm_castsi128_pd(xL));
            }

            template <class A>
            XSIMD_INLINE batch<double, A> fast_cast(batch<int64_t, A> const& x, batch<double, A> const&, requires_arch<altivec>) noexcept
            {
                // from https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx
                // adapted to altivec
                __m128i xH = _mm_srai_epi32(x, 16);
                xH = _mm_and_si128(xH, _mm_setr_epi16(0x0000, 0x0000, 0xFFFF, 0xFFFF, 0x0000, 0x0000, 0xFFFF, 0xFFFF));
                xH = _mm_add_epi64(xH, _mm_castpd_si128(_mm_set1_pd(442721857769029238784.))); //  3*2^67
                __m128i mask = _mm_setr_epi16(0xFFFF, 0xFFFF, 0xFFFF, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0x0000);
                __m128i xL = _mm_or_si128(_mm_and_si128(mask, x), _mm_andnot_si128(mask, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)))); //  2^52
                __m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(442726361368656609280.)); //  3*2^67 + 2^52
                return _mm_add_pd(f, _mm_castsi128_pd(xL));
            }

            template <class A>
            XSIMD_INLINE batch<int32_t, A> fast_cast(batch<float, A> const& self, batch<int32_t, A> const&, requires_arch<altivec>) noexcept
            {
                return _mm_cvttps_epi32(self);
            }
        }
#endif

        // eq
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_cmpeq(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch_bool<T, A> eq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_cmpeq(self, other);
        }

        // first
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE T first(batch<T, A> const& self, requires_arch<altivec>) noexcept
        {
            return vec_extract(self, 0);
        }
#if 0

        // from_mask
        template <class A>
        XSIMD_INLINE batch_bool<float, A> from_mask(batch_bool<float, A> const&, uint64_t mask, requires_arch<altivec>) noexcept
        {
            alignas(A::alignment()) static const uint32_t lut[][4] = {
                { 0x00000000, 0x00000000, 0x00000000, 0x00000000 },
                { 0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000 },
                { 0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000 },
                { 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000 },
                { 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000 },
                { 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000 },
                { 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000 },
                { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000 },
                { 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF },
                { 0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF },
                { 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF },
                { 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF },
                { 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF },
                { 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF },
                { 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF },
                { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF },
            };
            assert(!(mask & ~0xFul) && "inbound mask");
            return _mm_castsi128_ps(_mm_load_si128((const __m128i*)lut[mask]));
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> from_mask(batch_bool<double, A> const&, uint64_t mask, requires_arch<altivec>) noexcept
        {
            alignas(A::alignment()) static const uint64_t lut[][4] = {
                { 0x0000000000000000ul, 0x0000000000000000ul },
                { 0xFFFFFFFFFFFFFFFFul, 0x0000000000000000ul },
                { 0x0000000000000000ul, 0xFFFFFFFFFFFFFFFFul },
                { 0xFFFFFFFFFFFFFFFFul, 0xFFFFFFFFFFFFFFFFul },
            };
            assert(!(mask & ~0x3ul) && "inbound mask");
            return _mm_castsi128_pd(_mm_load_si128((const __m128i*)lut[mask]));
        }
        template <class T, class A, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch_bool<T, A> from_mask(batch_bool<T, A> const&, uint64_t mask, requires_arch<altivec>) noexcept
        {
            alignas(A::alignment()) static const uint64_t lut64[] = {
                0x0000000000000000,
                0x000000000000FFFF,
                0x00000000FFFF0000,
                0x00000000FFFFFFFF,
                0x0000FFFF00000000,
                0x0000FFFF0000FFFF,
                0x0000FFFFFFFF0000,
                0x0000FFFFFFFFFFFF,
                0xFFFF000000000000,
                0xFFFF00000000FFFF,
                0xFFFF0000FFFF0000,
                0xFFFF0000FFFFFFFF,
                0xFFFFFFFF00000000,
                0xFFFFFFFF0000FFFF,
                0xFFFFFFFFFFFF0000,
                0xFFFFFFFFFFFFFFFF,
            };
            alignas(A::alignment()) static const uint32_t lut32[] = {
                0x00000000,
                0x000000FF,
                0x0000FF00,
                0x0000FFFF,
                0x00FF0000,
                0x00FF00FF,
                0x00FFFF00,
                0x00FFFFFF,
                0xFF000000,
                0xFF0000FF,
                0xFF00FF00,
                0xFF00FFFF,
                0xFFFF0000,
                0xFFFF00FF,
                0xFFFFFF00,
                0xFFFFFFFF,
            };
            XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
            {
                assert(!(mask & ~0xFFFF) && "inbound mask");
                return _mm_setr_epi32(lut32[mask & 0xF], lut32[(mask >> 4) & 0xF], lut32[(mask >> 8) & 0xF], lut32[mask >> 12]);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
            {
                assert(!(mask & ~0xFF) && "inbound mask");
                return _mm_set_epi64x(lut64[mask >> 4], lut64[mask & 0xF]);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                return _mm_castps_si128(from_mask(batch_bool<float, A> {}, mask, altivec {}));
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
            {
                return _mm_castpd_si128(from_mask(batch_bool<double, A> {}, mask, altivec {}));
            }
        }
#endif
        // ge
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch_bool<T, A> ge(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_cmpge(self, other);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> ge(batch<double, A> const& self, batch<double, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_cmpge(self, other);
        }

        // gt
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_cmpgt(self, other);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> gt(batch<double, A> const& self, batch<double, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_cmpgt(self, other);
        }

#if 0

        // haddp
        template <class A>
        XSIMD_INLINE batch<float, A> haddp(batch<float, A> const* row, requires_arch<altivec>) noexcept
        {
            __m128 tmp0 = _mm_unpacklo_ps(row[0], row[1]);
            __m128 tmp1 = _mm_unpackhi_ps(row[0], row[1]);
            __m128 tmp2 = _mm_unpackhi_ps(row[2], row[3]);
            tmp0 = _mm_add_ps(tmp0, tmp1);
            tmp1 = _mm_unpacklo_ps(row[2], row[3]);
            tmp1 = _mm_add_ps(tmp1, tmp2);
            tmp2 = _mm_movehl_ps(tmp1, tmp0);
            tmp0 = _mm_movelh_ps(tmp0, tmp1);
            return _mm_add_ps(tmp0, tmp2);
        }
#endif

        // incr_if
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> incr_if(batch<T, A> const& self, batch_bool<T, A> const& mask, requires_arch<altivec>) noexcept
        {
            return self - batch<T, A>(mask.data);
        }

        // insert
        template <class A, class T, size_t I, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> insert(batch<T, A> const& self, T val, index<I> pos, requires_arch<altivec>) noexcept
        {
            return vec_insert(val, self, pos);
        }

        // isnan
        template <class A>
        XSIMD_INLINE batch_bool<float, A> isnan(batch<float, A> const& self, requires_arch<altivec>) noexcept
        {
            return ~vec_cmpeq(self, self);
        }

        // load_aligned
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> load_aligned(T const* mem, convert<T>, requires_arch<altivec>) noexcept
        {
            return vec_ld(0, mem);
        }

        // load_unaligned
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> load_unaligned(T const* mem, convert<T>, requires_arch<altivec>) noexcept
        {
            return batch<T, A>(*(typename batch<T, A>::register_type)mem);
        }

#if 0
        // load_complex
        namespace detail
        {
            // Redefine these methods in the SSE-based archs if required
            template <class A>
            XSIMD_INLINE batch<std::complex<float>, A> load_complex(batch<float, A> const& hi, batch<float, A> const& lo, requires_arch<altivec>) noexcept
            {
                return { _mm_shuffle_ps(hi, lo, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_ps(hi, lo, _MM_SHUFFLE(3, 1, 3, 1)) };
            }
            template <class A>
            XSIMD_INLINE batch<std::complex<double>, A> load_complex(batch<double, A> const& hi, batch<double, A> const& lo, requires_arch<altivec>) noexcept
            {
                return { _mm_shuffle_pd(hi, lo, _MM_SHUFFLE2(0, 0)), _mm_shuffle_pd(hi, lo, _MM_SHUFFLE2(1, 1)) };
            }
        }
#endif

        // le
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch_bool<T, A> le(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_cmple(self, other);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> le(batch<double, A> const& self, batch<double, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_cmple(self, other);
        }

        // lt
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch_bool<T, A> lt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_cmplt(self, other);
        }
        template <class A>
        XSIMD_INLINE batch_bool<double, A> lt(batch<double, A> const& self, batch<double, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_cmplt(self, other);
        }

#if 0



        /* compression table to turn 0b10 into 0b1,
         * 0b100010 into 0b101 etc
         */
        namespace detail
        {
            XSIMD_INLINE int mask_lut(uint64_t mask)
            {
                // clang-format off
                static const int mask_lut[256] = {
                  0x0, 0x0, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2, 0x0, 0x3, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x4, 0x0, 0x5, 0x0, 0x0, 0x0, 0x0, 0x0, 0x6, 0x0, 0x7, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x8, 0x0, 0x9, 0x0, 0x0, 0x0, 0x0, 0x0, 0xA, 0x0, 0xB, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0xC, 0x0, 0xD, 0x0, 0x0, 0x0, 0x0, 0x0, 0xE, 0x0, 0xF, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                };
                // clang-format on
                return mask_lut[mask & 0xAA];
            }
        }

        // mask
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE uint64_t mask(batch_bool<T, A> const& self, requires_arch<altivec>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
            {
                return _mm_movemask_epi8(self);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 2)
            {
                uint64_t mask8 = _mm_movemask_epi8(self);
                return detail::mask_lut(mask8) | (detail::mask_lut(mask8 >> 8) << 4);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                return _mm_movemask_ps(_mm_castsi128_ps(self));
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
            {
                return _mm_movemask_pd(_mm_castsi128_pd(self));
            }
            else
            {
                assert(false && "unsupported arch/op combination");
                return {};
            }
        }
        template <class A>
        XSIMD_INLINE uint64_t mask(batch_bool<float, A> const& self, requires_arch<altivec>) noexcept
        {
            return _mm_movemask_ps(self);
        }

        template <class A>
        XSIMD_INLINE uint64_t mask(batch_bool<double, A> const& self, requires_arch<altivec>) noexcept
        {
            return _mm_movemask_pd(self);
        }

#endif

        // max
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_max(self, other);
        }

        // min
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_min(self, other);
        }

        // mul
        template <class A, class T, typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_mul(self, other);
        }
#if 0

        // nearbyint_as_int
        template <class A>
        XSIMD_INLINE batch<int32_t, A> nearbyint_as_int(batch<float, A> const& self,
                                                        requires_arch<altivec>) noexcept
        {
            return _mm_cvtps_epi32(self);
        }
#endif

        // neg
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> neg(batch<T, A> const& self, requires_arch<altivec>) noexcept
        {
            return vec_neg(self);
        }

        // neq
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return ~vec_cmpeq(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch_bool<T, A> neq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return ~vec_cmpeq(self, other);
        }

        // reciprocal
        template <class A>
        XSIMD_INLINE batch<float, A> reciprocal(batch<float, A> const& self,
                                                kernel::requires_arch<altivec>)
        {
            return vec_re(self);
        }

#if 0
        // reduce_add
        template <class A>
        XSIMD_INLINE float reduce_add(batch<float, A> const& self, requires_arch<altivec>) noexcept
        {
            __m128 tmp0 = _mm_add_ps(self, _mm_movehl_ps(self, self));
            __m128 tmp1 = _mm_add_ss(tmp0, _mm_shuffle_ps(tmp0, tmp0, 1));
            return _mm_cvtss_f32(tmp1);
        }

        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE T reduce_add(batch<T, A> const& self, requires_arch<altivec>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 4)
            {
                __m128i tmp1 = _mm_shuffle_epi32(self, 0x0E);
                __m128i tmp2 = _mm_add_epi32(self, tmp1);
                __m128i tmp3 = _mm_shuffle_epi32(tmp2, 0x01);
                __m128i tmp4 = _mm_add_epi32(tmp2, tmp3);
                return _mm_cvtsi128_si32(tmp4);
            }
            else XSIMD_IF_CONSTEXPR(sizeof(T) == 8)
            {
                __m128i tmp1 = _mm_shuffle_epi32(self, 0x0E);
                __m128i tmp2 = _mm_add_epi64(self, tmp1);
#if defined(__x86_64__)
                return _mm_cvtsi128_si64(tmp2);
#else
                __m128i m;
                _mm_storel_epi64(&m, tmp2);
                int64_t i;
                std::memcpy(&i, &m, sizeof(i));
                return i;
#endif
            }
            else
            {
                return hadd(self, common {});
            }
        }

        template <class A>
        XSIMD_INLINE double reduce_add(batch<double, A> const& self, requires_arch<altivec>) noexcept
        {
            return _mm_cvtsd_f64(_mm_add_sd(self, _mm_unpackhi_pd(self, self)));
        }

        // reduce_max
        template <class A, class T, class _ = typename std::enable_if<(sizeof(T) <= 2), void>::type>
        XSIMD_INLINE T reduce_max(batch<T, A> const& self, requires_arch<altivec>) noexcept
        {
            constexpr auto mask0 = detail::shuffle(2, 3, 0, 0);
            batch<T, A> step0 = _mm_shuffle_epi32(self, mask0);
            batch<T, A> acc0 = max(self, step0);

            constexpr auto mask1 = detail::shuffle(1, 0, 0, 0);
            batch<T, A> step1 = _mm_shuffle_epi32(acc0, mask1);
            batch<T, A> acc1 = max(acc0, step1);

            constexpr auto mask2 = detail::shuffle(1, 0, 0, 0);
            batch<T, A> step2 = _mm_shufflelo_epi16(acc1, mask2);
            batch<T, A> acc2 = max(acc1, step2);
            if (sizeof(T) == 2)
                return first(acc2, A {});
            batch<T, A> step3 = bitwise_cast<T>(bitwise_cast<uint16_t>(acc2) >> 8);
            batch<T, A> acc3 = max(acc2, step3);
            return first(acc3, A {});
        }

        // reduce_min
        template <class A, class T, class _ = typename std::enable_if<(sizeof(T) <= 2), void>::type>
        XSIMD_INLINE T reduce_min(batch<T, A> const& self, requires_arch<altivec>) noexcept
        {
            constexpr auto mask0 = detail::shuffle(2, 3, 0, 0);
            batch<T, A> step0 = _mm_shuffle_epi32(self, mask0);
            batch<T, A> acc0 = min(self, step0);

            constexpr auto mask1 = detail::shuffle(1, 0, 0, 0);
            batch<T, A> step1 = _mm_shuffle_epi32(acc0, mask1);
            batch<T, A> acc1 = min(acc0, step1);

            constexpr auto mask2 = detail::shuffle(1, 0, 0, 0);
            batch<T, A> step2 = _mm_shufflelo_epi16(acc1, mask2);
            batch<T, A> acc2 = min(acc1, step2);
            if (sizeof(T) == 2)
                return first(acc2, A {});
            batch<T, A> step3 = bitwise_cast<T>(bitwise_cast<uint16_t>(acc2) >> 8);
            batch<T, A> acc3 = min(acc2, step3);
            return first(acc3, A {});
        }
#endif

        // rsqrt
        template <class A>
        XSIMD_INLINE batch<float, A> rsqrt(batch<float, A> const& val, requires_arch<altivec>) noexcept
        {
            return vec_rsqrt(val);
        }

        // select
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<altivec>) noexcept
        {
            return vec_sel(true_br, false_br, cond);
        }
        template <class A, class T, bool... Values, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> select(batch_bool_constant<T, A, Values...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<altivec>) noexcept
        {
            return select(batch_bool<T, A> { Values... }, true_br, false_br, altivec {});
        }
#if 0

        // shuffle
        template <class A, class ITy, ITy I0, ITy I1, ITy I2, ITy I3>
        XSIMD_INLINE batch<float, A> shuffle(batch<float, A> const& x, batch<float, A> const& y, batch_constant<ITy, A, I0, I1, I2, I3> mask, requires_arch<altivec>) noexcept
        {
            constexpr uint32_t smask = detail::mod_shuffle(I0, I1, I2, I3);
            // shuffle within lane
            if (I0 < 4 && I1 < 4 && I2 >= 4 && I3 >= 4)
                return _mm_shuffle_ps(x, y, smask);

            // shuffle within opposite lane
            if (I0 >= 4 && I1 >= 4 && I2 < 4 && I3 < 4)
                return _mm_shuffle_ps(y, x, smask);
            return shuffle(x, y, mask, common {});
        }

        template <class A, class ITy, ITy I0, ITy I1>
        XSIMD_INLINE batch<double, A> shuffle(batch<double, A> const& x, batch<double, A> const& y, batch_constant<ITy, A, I0, I1> mask, requires_arch<altivec>) noexcept
        {
            constexpr uint32_t smask = detail::mod_shuffle(I0, I1);
            // shuffle within lane
            if (I0 < 2 && I1 >= 2)
                return _mm_shuffle_pd(x, y, smask);

            // shuffle within opposite lane
            if (I0 >= 2 && I1 < 2)
                return _mm_shuffle_pd(y, x, smask);
            return shuffle(x, y, mask, common {});
        }
#endif

        // sqrt
        template <class A>
        XSIMD_INLINE batch<float, A> sqrt(batch<float, A> const& val, requires_arch<altivec>) noexcept
        {
            return vec_sqrt(val);
        }
#if 0

        // slide_left
        template <size_t N, class A, class T>
        XSIMD_INLINE batch<T, A> slide_left(batch<T, A> const& x, requires_arch<altivec>) noexcept
        {
            return _mm_slli_si128(x, N);
        }

        // slide_right
        template <size_t N, class A, class T>
        XSIMD_INLINE batch<T, A> slide_right(batch<T, A> const& x, requires_arch<altivec>) noexcept
        {
            return _mm_srli_si128(x, N);
        }

#endif
        // sadd
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> sadd(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_adds(self, other);
        }

        // set
        template <class A, class T, class... Values>
        XSIMD_INLINE batch<float, A> set(batch<T, A> const&, requires_arch<altivec>, Values... values) noexcept
        {
            static_assert(sizeof...(Values) == batch<T, A>::size, "consistent init");
            return typename batch<T, A>::register_type { values... };
        }

        // ssub

        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> ssub(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            XSIMD_IF_CONSTEXPR(sizeof(T) == 1)
            {
                return vec_subs(self, other);
            }
            else
            {
                return ssub(self, other, common {});
            }
        }

        // store_aligned
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE void store_aligned(T* mem, batch<T, A> const& self, requires_arch<altivec>) noexcept
        {
            return vec_st(self.data, 0, mem);
        }

        // store_unaligned
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE void store_unaligned(T* mem, batch<T, A> const& self, requires_arch<altivec>) noexcept
        {
            *(typename batch<T, A>::register_type)mem = self.data;
        }

        // sub
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> sub(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_sub(self, other);
        }

#if 0
        // swizzle

        template <class A, uint32_t V0, uint32_t V1, uint32_t V2, uint32_t V3>
        XSIMD_INLINE batch<float, A> swizzle(batch<float, A> const& self, batch_constant<uint32_t, A, V0, V1, V2, V3>, requires_arch<altivec>) noexcept
        {
            constexpr uint32_t index = detail::shuffle(V0, V1, V2, V3);
            return _mm_shuffle_ps(self, self, index);
        }

        template <class A, uint64_t V0, uint64_t V1>
        XSIMD_INLINE batch<double, A> swizzle(batch<double, A> const& self, batch_constant<uint64_t, A, V0, V1>, requires_arch<altivec>) noexcept
        {
            constexpr uint32_t index = detail::shuffle(V0, V1);
            return _mm_shuffle_pd(self, self, index);
        }

        template <class A, uint64_t V0, uint64_t V1>
        XSIMD_INLINE batch<uint64_t, A> swizzle(batch<uint64_t, A> const& self, batch_constant<uint64_t, A, V0, V1>, requires_arch<altivec>) noexcept
        {
            constexpr uint32_t index = detail::shuffle(2 * V0, 2 * V0 + 1, 2 * V1, 2 * V1 + 1);
            return _mm_shuffle_epi32(self, index);
        }

        template <class A, uint64_t V0, uint64_t V1>
        XSIMD_INLINE batch<int64_t, A> swizzle(batch<int64_t, A> const& self, batch_constant<uint64_t, A, V0, V1> mask, requires_arch<altivec>) noexcept
        {
            return bitwise_cast<int64_t>(swizzle(bitwise_cast<uint64_t>(self), mask, altivec {}));
        }

        template <class A, uint32_t V0, uint32_t V1, uint32_t V2, uint32_t V3>
        XSIMD_INLINE batch<uint32_t, A> swizzle(batch<uint32_t, A> const& self, batch_constant<uint32_t, A, V0, V1, V2, V3>, requires_arch<altivec>) noexcept
        {
            constexpr uint32_t index = detail::shuffle(V0, V1, V2, V3);
            return _mm_shuffle_epi32(self, index);
        }

        template <class A, uint32_t V0, uint32_t V1, uint32_t V2, uint32_t V3>
        XSIMD_INLINE batch<int32_t, A> swizzle(batch<int32_t, A> const& self, batch_constant<uint32_t, A, V0, V1, V2, V3> mask, requires_arch<altivec>) noexcept
        {
            return bitwise_cast<int32_t>(swizzle(bitwise_cast<uint32_t>(self), mask, altivec {}));
        }

        template <class A, uint16_t V0, uint16_t V1, uint16_t V2, uint16_t V3, uint16_t V4, uint16_t V5, uint16_t V6, uint16_t V7>
        XSIMD_INLINE batch<uint16_t, A> swizzle(batch<uint16_t, A> const& self, batch_constant<uint16_t, A, V0, V1, V2, V3, V4, V5, V6, V7>, requires_arch<altivec>) noexcept
        {
            // permute within each lane
            constexpr auto mask_lo = detail::mod_shuffle(V0, V1, V2, V3);
            constexpr auto mask_hi = detail::mod_shuffle(V4, V5, V6, V7);
            __m128i lo = _mm_shufflelo_epi16(self, mask_lo);
            __m128i hi = _mm_shufflehi_epi16(self, mask_hi);

            __m128i lo_lo = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(lo), _mm_castsi128_pd(lo), _MM_SHUFFLE2(0, 0)));
            __m128i hi_hi = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(hi), _mm_castsi128_pd(hi), _MM_SHUFFLE2(1, 1)));

            // mask to choose the right lane
            batch_bool_constant<uint16_t, A, (V0 < 4), (V1 < 4), (V2 < 4), (V3 < 4), (V4 < 4), (V5 < 4), (V6 < 4), (V7 < 4)> blend_mask;

            // blend the two permutes
            return select(blend_mask, batch<uint16_t, A>(lo_lo), batch<uint16_t, A>(hi_hi));
        }

        template <class A, uint16_t V0, uint16_t V1, uint16_t V2, uint16_t V3, uint16_t V4, uint16_t V5, uint16_t V6, uint16_t V7>
        XSIMD_INLINE batch<int16_t, A> swizzle(batch<int16_t, A> const& self, batch_constant<uint16_t, A, V0, V1, V2, V3, V4, V5, V6, V7> mask, requires_arch<altivec>) noexcept
        {
            return bitwise_cast<int16_t>(swizzle(bitwise_cast<uint16_t>(self), mask, altivec {}));
        }

        // transpose
        template <class A>
        XSIMD_INLINE void transpose(batch<float, A>* matrix_begin, batch<float, A>* matrix_end, requires_arch<altivec>) noexcept
        {
            assert((matrix_end - matrix_begin == batch<float, A>::size) && "correctly sized matrix");
            (void)matrix_end;
            auto r0 = matrix_begin[0], r1 = matrix_begin[1], r2 = matrix_begin[2], r3 = matrix_begin[3];
            _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
            matrix_begin[0] = r0;
            matrix_begin[1] = r1;
            matrix_begin[2] = r2;
            matrix_begin[3] = r3;
        }
        template <class A>
        XSIMD_INLINE void transpose(batch<uint32_t, A>* matrix_begin, batch<uint32_t, A>* matrix_end, requires_arch<altivec>) noexcept
        {
            transpose(reinterpret_cast<batch<float, A>*>(matrix_begin), reinterpret_cast<batch<float, A>*>(matrix_end), A {});
        }
        template <class A>
        XSIMD_INLINE void transpose(batch<int32_t, A>* matrix_begin, batch<int32_t, A>* matrix_end, requires_arch<altivec>) noexcept
        {
            transpose(reinterpret_cast<batch<float, A>*>(matrix_begin), reinterpret_cast<batch<float, A>*>(matrix_end), A {});
        }

        template <class A>
        XSIMD_INLINE void transpose(batch<double, A>* matrix_begin, batch<double, A>* matrix_end, requires_arch<altivec>) noexcept
        {
            assert((matrix_end - matrix_begin == batch<double, A>::size) && "correctly sized matrix");
            (void)matrix_end;
            auto r0 = matrix_begin[0], r1 = matrix_begin[1];
            matrix_begin[0] = _mm_unpacklo_pd(r0, r1);
            matrix_begin[1] = _mm_unpackhi_pd(r0, r1);
        }
        template <class A>
        XSIMD_INLINE void transpose(batch<uint64_t, A>* matrix_begin, batch<uint64_t, A>* matrix_end, requires_arch<altivec>) noexcept
        {
            transpose(reinterpret_cast<batch<double, A>*>(matrix_begin), reinterpret_cast<batch<double, A>*>(matrix_end), A {});
        }
        template <class A>
        XSIMD_INLINE void transpose(batch<int64_t, A>* matrix_begin, batch<int64_t, A>* matrix_end, requires_arch<altivec>) noexcept
        {
            transpose(reinterpret_cast<batch<double, A>*>(matrix_begin), reinterpret_cast<batch<double, A>*>(matrix_end), A {});
        }

#endif
        // zip_hi
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> zip_hi(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_merge_hi(self, other);
        }

        // zip_lo
        template <class A, class T, class = typename std::enable_if<std::is_scalar<T>::value, void>::type>
        XSIMD_INLINE batch<T, A> zip_lo(batch<T, A> const& self, batch<T, A> const& other, requires_arch<altivec>) noexcept
        {
            return vec_mergel(self, other);
        }
    }
}

#endif
