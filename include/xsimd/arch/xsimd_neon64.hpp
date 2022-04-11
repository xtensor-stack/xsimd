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

#ifndef XSIMD_NEON64_HPP
#define XSIMD_NEON64_HPP

#include <complex>
#include <cstddef>
#include <tuple>

#include "../types/xsimd_neon64_register.hpp"
#include "../types/xsimd_utils.hpp"

namespace xsimd
{
    template <class batch_type, bool... Values>
    struct batch_bool_constant;

    namespace kernel
    {
        using namespace types;
        /*******
         * all *
         *******/

        template <class A, class T, detail::enable_sized_t<T, 4> = 0>
        inline bool all(batch_bool<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return vminvq_u32(arg) == ~0U;
        }

        template <class A, class T, detail::enable_sized_t<T, 1> = 0>
        inline bool all(batch_bool<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return all(batch_bool<uint32_t, A>(vreinterpretq_u32_u8(arg)), neon64 {});
        }

        template <class A, class T, detail::enable_sized_t<T, 2> = 0>
        inline bool all(batch_bool<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return all(batch_bool<uint32_t, A>(vreinterpretq_u32_u16(arg)), neon64 {});
        }

        template <class A, class T, detail::enable_sized_t<T, 8> = 0>
        inline bool all(batch_bool<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return all(batch_bool<uint32_t, A>(vreinterpretq_u32_u64(arg)), neon64 {});
        }

        /*******
         * any *
         *******/

        template <class A, class T, detail::enable_sized_t<T, 4> = 0>
        inline bool any(batch_bool<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return vmaxvq_u32(arg) != 0;
        }

        template <class A, class T, detail::enable_sized_t<T, 1> = 0>
        inline bool any(batch_bool<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return any(batch_bool<uint32_t, A>(vreinterpretq_u32_u8(arg)), neon64 {});
        }

        template <class A, class T, detail::enable_sized_t<T, 2> = 0>
        inline bool any(batch_bool<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return any(batch_bool<uint32_t, A>(vreinterpretq_u32_u16(arg)), neon64 {});
        }

        template <class A, class T, detail::enable_sized_t<T, 8> = 0>
        inline bool any(batch_bool<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return any(batch_bool<uint32_t, A>(vreinterpretq_u32_u64(arg)), neon64 {});
        }

        /*************
         * broadcast *
         *************/

        // Required to avoid ambiguous call
        template <class A, class T>
        inline batch<T, A> broadcast(T val, requires_arch<neon64>) noexcept
        {
            return broadcast<neon64>(val, neon {});
        }

        template <class A>
        inline batch<double, A> broadcast(double val, requires_arch<neon64>) noexcept
        {
            return vdupq_n_f64(val);
        }

        /*******
         * set *
         *******/

        template <class A>
        inline batch<double, A> set(batch<double, A> const&, requires_arch<neon64>, double d0, double d1) noexcept
        {
            return float64x2_t { d0, d1 };
        }

        template <class A>
        inline batch_bool<double, A> set(batch_bool<double, A> const&, requires_arch<neon64>, bool b0, bool b1) noexcept
        {
            using register_type = typename batch_bool<double, A>::register_type;
            using unsigned_type = as_unsigned_integer_t<double>;
            return register_type { static_cast<unsigned_type>(b0 ? -1LL : 0LL),
                                   static_cast<unsigned_type>(b1 ? -1LL : 0LL) };
        }

        /*************
         * from_bool *
         *************/

        template <class A>
        inline batch<double, A> from_bool(batch_bool<double, A> const& arg, requires_arch<neon64>) noexcept
        {
            return vreinterpretq_f64_u64(vandq_u64(arg, vreinterpretq_u64_f64(vdupq_n_f64(1.))));
        }

        /********
         * load *
         ********/

        template <class A>
        inline batch<double, A> load_aligned(double const* src, convert<double>, requires_arch<neon64>) noexcept
        {
            return vld1q_f64(src);
        }

        template <class A>
        inline batch<double, A> load_unaligned(double const* src, convert<double>, requires_arch<neon64>) noexcept
        {
            return load_aligned<A>(src, convert<double>(), A {});
        }

        /*********
         * store *
         *********/

        template <class A>
        inline void store_aligned(double* dst, batch<double, A> const& src, requires_arch<neon64>) noexcept
        {
            vst1q_f64(dst, src);
        }

        template <class A>
        inline void store_unaligned(double* dst, batch<double, A> const& src, requires_arch<neon64>) noexcept
        {
            return store_aligned<A>(dst, src, A {});
        }

        /****************
         * load_complex *
         ****************/

        template <class A>
        inline batch<std::complex<double>, A> load_complex_aligned(std::complex<double> const* mem, convert<std::complex<double>>, requires_arch<neon64>) noexcept
        {
            using real_batch = batch<double, A>;
            const double* buf = reinterpret_cast<const double*>(mem);
            float64x2x2_t tmp = vld2q_f64(buf);
            real_batch real = tmp.val[0],
                       imag = tmp.val[1];
            return batch<std::complex<double>, A> { real, imag };
        }

        template <class A>
        inline batch<std::complex<double>, A> load_complex_unaligned(std::complex<double> const* mem, convert<std::complex<double>> cvt, requires_arch<neon64>) noexcept
        {
            return load_complex_aligned<A>(mem, cvt, A {});
        }

        /*****************
         * store_complex *
         *****************/

        template <class A>
        inline void store_complex_aligned(std::complex<double>* dst, batch<std::complex<double>, A> const& src, requires_arch<neon64>) noexcept
        {
            float64x2x2_t tmp;
            tmp.val[0] = src.real();
            tmp.val[1] = src.imag();
            double* buf = reinterpret_cast<double*>(dst);
            vst2q_f64(buf, tmp);
        }

        template <class A>
        inline void store_complex_unaligned(std::complex<double>* dst, batch<std::complex<double>, A> const& src, requires_arch<neon64>) noexcept
        {
            store_complex_aligned(dst, src, A {});
        }

        /*******
         * neg *
         *******/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline batch<T, A> neg(batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vreinterpretq_u64_s64(vnegq_s64(vreinterpretq_s64_u64(rhs)));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline batch<T, A> neg(batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vnegq_s64(rhs);
        }

        template <class A>
        inline batch<double, A> neg(batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vnegq_f64(rhs);
        }

        /*******
         * add *
         *******/

        template <class A>
        inline batch<double, A> add(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vaddq_f64(lhs, rhs);
        }

        /********
         * sadd *
         ********/

        template <class A>
        inline batch<double, A> sadd(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return add(lhs, rhs, neon64 {});
        }

        /*******
         * sub *
         *******/

        template <class A>
        inline batch<double, A> sub(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vsubq_f64(lhs, rhs);
        }

        /********
         * ssub *
         ********/

        template <class A>
        inline batch<double, A> ssub(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return sub(lhs, rhs, neon64 {});
        }

        /*******
         * mul *
         *******/

        template <class A>
        inline batch<double, A> mul(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vmulq_f64(lhs, rhs);
        }

        /*******
         * div *
         *******/

#if defined(XSIMD_FAST_INTEGER_DIVISION)
        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline batch<T, A> div(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcvtq_u64_f64(vcvtq_f64_u64(lhs) / vcvtq_f64_u64(rhs));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline batch<T, A> div(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcvtq_s64_f64(vcvtq_f64_s64(lhs) / vcvtq_f64_s64(rhs));
        }
#endif
        template <class A>
        inline batch<double, A> div(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vdivq_f64(lhs, rhs);
        }

        /******
         * eq *
         ******/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline batch_bool<T, A> eq(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vceqq_u64(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline batch_bool<T, A> eq(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vceqq_s64(lhs, rhs);
        }

        template <class A>
        inline batch_bool<double, A> eq(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vceqq_f64(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline batch_bool<T, A> eq(batch_bool<T, A> const& lhs, batch_bool<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vceqq_u64(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline batch_bool<T, A> eq(batch_bool<T, A> const& lhs, batch_bool<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vceqq_u64(lhs, rhs);
        }

        template <class A>
        inline batch_bool<double, A> eq(batch_bool<double, A> const& lhs, batch_bool<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vceqq_u64(lhs, rhs);
        }

        /*************
         * fast_cast *
         *************/
        namespace detail
        {
            template <class A>
            inline batch<double, A> fast_cast(batch<int64_t, A> const& x, batch<double, A> const&, requires_arch<neon64>) noexcept
            {
                return vcvtq_f64_s64(x);
            }

            template <class A>
            inline batch<double, A> fast_cast(batch<uint64_t, A> const& x, batch<double, A> const&, requires_arch<neon64>) noexcept
            {
                return vcvtq_f64_u64(x);
            }

            template <class A>
            inline batch<int64_t, A> fast_cast(batch<double, A> const& x, batch<int64_t, A> const&, requires_arch<neon64>) noexcept
            {
                return vcvtq_s64_f64(x);
            }

            template <class A>
            inline batch<uint64_t, A> fast_cast(batch<double, A> const& x, batch<uint64_t, A> const&, requires_arch<neon64>) noexcept
            {
                return vcvtq_u64_f64(x);
            }

        }

        /******
         * lt *
         ******/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline batch_bool<T, A> lt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcltq_u64(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline batch_bool<T, A> lt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcltq_s64(lhs, rhs);
        }

        template <class A>
        inline batch_bool<double, A> lt(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcltq_f64(lhs, rhs);
        }

        /******
         * le *
         ******/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline batch_bool<T, A> le(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcleq_u64(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline batch_bool<T, A> le(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcleq_s64(lhs, rhs);
        }

        template <class A>
        inline batch_bool<double, A> le(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcleq_f64(lhs, rhs);
        }

        /******
         * gt *
         ******/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline batch_bool<T, A> gt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcgtq_u64(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline batch_bool<T, A> gt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcgtq_s64(lhs, rhs);
        }

        template <class A>
        inline batch_bool<double, A> gt(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcgtq_f64(lhs, rhs);
        }

        /******
         * ge *
         ******/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline batch_bool<T, A> ge(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcgeq_u64(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline batch_bool<T, A> ge(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcgeq_s64(lhs, rhs);
        }

        template <class A>
        inline batch_bool<double, A> ge(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vcgeq_f64(lhs, rhs);
        }

        /***************
         * bitwise_and *
         ***************/

        template <class A>
        inline batch<double, A> bitwise_and(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(lhs),
                                                   vreinterpretq_u64_f64(rhs)));
        }

        template <class A>
        inline batch_bool<double, A> bitwise_and(batch_bool<double, A> const& lhs, batch_bool<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vandq_u64(lhs, rhs);
        }

        /**************
         * bitwise_or *
         **************/

        template <class A>
        inline batch<double, A> bitwise_or(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(lhs),
                                                   vreinterpretq_u64_f64(rhs)));
        }

        template <class A>
        inline batch_bool<double, A> bitwise_or(batch_bool<double, A> const& lhs, batch_bool<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vorrq_u64(lhs, rhs);
        }

        /***************
         * bitwise_xor *
         ***************/

        template <class A>
        inline batch<double, A> bitwise_xor(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(lhs),
                                                   vreinterpretq_u64_f64(rhs)));
        }

        template <class A>
        inline batch_bool<double, A> bitwise_xor(batch_bool<double, A> const& lhs, batch_bool<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return veorq_u64(lhs, rhs);
        }

        /*******
         * neq *
         *******/

        template <class A>
        inline batch_bool<double, A> neq(batch_bool<double, A> const& lhs, batch_bool<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return bitwise_xor(lhs, rhs, A {});
        }

        /***************
         * bitwise_not *
         ***************/

        template <class A>
        inline batch<double, A> bitwise_not(batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vreinterpretq_f64_u32(vmvnq_u32(vreinterpretq_u32_f64(rhs)));
        }

        template <class A>
        inline batch_bool<double, A> bitwise_not(batch_bool<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return detail::bitwise_not_u64(rhs);
        }

        /******************
         * bitwise_andnot *
         ******************/

        template <class A>
        inline batch<double, A> bitwise_andnot(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(lhs),
                                                   vreinterpretq_u64_f64(rhs)));
        }

        template <class A>
        inline batch_bool<double, A> bitwise_andnot(batch_bool<double, A> const& lhs, batch_bool<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vbicq_u64(lhs, rhs);
        }

        /*******
         * min *
         *******/

        template <class A>
        inline batch<double, A> min(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vminq_f64(lhs, rhs);
        }

        /*******
         * max *
         *******/

        template <class A>
        inline batch<double, A> max(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vmaxq_f64(lhs, rhs);
        }

        /*******
         * abs *
         *******/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline batch<T, A> abs(batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return rhs;
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline batch<T, A> abs(batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vabsq_s64(rhs);
        }

        template <class A>
        inline batch<double, A> abs(batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vabsq_f64(rhs);
        }

        /**************
         * reciprocal *
         **************/

        template <class A>
        inline batch<double, A>
        reciprocal(const batch<double, A>& x,
                   kernel::requires_arch<neon64>) noexcept
        {
            return vrecpeq_f64(x);
        }

        /********
         * rsqrt *
         ********/

        template <class A>
        inline batch<double, A> rsqrt(batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vrsqrteq_f64(rhs);
        }

        /********
         * sqrt *
         ********/

        template <class A>
        inline batch<double, A> sqrt(batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vsqrtq_f64(rhs);
        }

        /********************
         * Fused operations *
         ********************/

#ifdef __ARM_FEATURE_FMA
        template <class A>
        inline batch<double, A> fma(batch<double, A> const& x, batch<double, A> const& y, batch<double, A> const& z, requires_arch<neon64>) noexcept
        {
            return vfmaq_f64(z, x, y);
        }

        template <class A>
        inline batch<double, A> fms(batch<double, A> const& x, batch<double, A> const& y, batch<double, A> const& z, requires_arch<neon64>) noexcept
        {
            return vfmaq_f64(-z, x, y);
        }
#endif

        /********
         * hadd *
         ********/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 1> = 0>
        inline typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return vaddvq_u8(arg);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 1> = 0>
        inline typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return vaddvq_s8(arg);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 2> = 0>
        inline typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return vaddvq_u16(arg);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 2> = 0>
        inline typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return vaddvq_s16(arg);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 4> = 0>
        inline typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return vaddvq_u32(arg);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 4> = 0>
        inline typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return vaddvq_s32(arg);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return vaddvq_u64(arg);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon64>) noexcept
        {
            return vaddvq_s64(arg);
        }

        template <class A>
        inline double hadd(batch<double, A> const& arg, requires_arch<neon64>) noexcept
        {
            return vaddvq_f64(arg);
        }

        /*********
         * haddp *
         *********/

        template <class A>
        inline batch<double, A> haddp(const batch<double, A>* row, requires_arch<neon64>) noexcept
        {
            return vpaddq_f64(row[0], row[1]);
        }

        /**********
         * select *
         **********/

        template <class A>
        inline batch<double, A> select(batch_bool<double, A> const& cond, batch<double, A> const& a, batch<double, A> const& b, requires_arch<neon64>) noexcept
        {
            return vbslq_f64(cond, a, b);
        }

        template <class A, bool... b>
        inline batch<double, A> select(batch_bool_constant<batch<double, A>, b...> const&,
                                       batch<double, A> const& true_br,
                                       batch<double, A> const& false_br,
                                       requires_arch<neon64>) noexcept
        {
            return select(batch_bool<double, A> { b... }, true_br, false_br, neon64 {});
        }
        /**********
         * zip_lo *
         **********/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline batch<T, A> zip_lo(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vzip1q_u64(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline batch<T, A> zip_lo(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vzip1q_s64(lhs, rhs);
        }

        template <class A>
        inline batch<double, A> zip_lo(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vzip1q_f64(lhs, rhs);
        }

        /**********
         * zip_hi *
         **********/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline batch<T, A> zip_hi(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vzip2q_u64(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline batch<T, A> zip_hi(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vzip2q_s64(lhs, rhs);
        }

        template <class A>
        inline batch<double, A> zip_hi(batch<double, A> const& lhs, batch<double, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vzip2q_f64(lhs, rhs);
        }

        /****************
         * extract_pair *
         ****************/

        namespace detail
        {
            template <class A, size_t I, size_t... Is>
            inline batch<double, A> extract_pair(batch<double, A> const& lhs, batch<double, A> const& rhs, std::size_t n,
                                                 ::xsimd::detail::index_sequence<I, Is...>) noexcept
            {
                if (n == I)
                {
                    return vextq_f64(rhs, lhs, I);
                }
                else
                {
                    return extract_pair(lhs, rhs, n, ::xsimd::detail::index_sequence<Is...>());
                }
            }
        }

        template <class A>
        inline batch<double, A> extract_pair(batch<double, A> const& lhs, batch<double, A> const& rhs, std::size_t n, requires_arch<neon64>) noexcept
        {
            constexpr std::size_t size = batch<double, A>::size;
            assert(0 <= n && n < size && "index in bounds");
            return detail::extract_pair(lhs, rhs, n, ::xsimd::detail::make_index_sequence<size>());
        }

        /******************
         * bitwise_rshift *
         ******************/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline batch<T, A> bitwise_rshift(batch<T, A> const& lhs, int n, requires_arch<neon64>) noexcept
        {
            return bitwise_rshift<A>(lhs, n, neon {});
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        inline batch<T, A> bitwise_rshift(batch<T, A> const& lhs, batch<as_signed_integer_t<T>, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vshlq_u64(lhs, vnegq_s64(rhs));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline batch<T, A> bitwise_rshift(batch<T, A> const& lhs, int n, requires_arch<neon64>) noexcept
        {
            return bitwise_rshift<A>(lhs, n, neon {});
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        inline batch<T, A> bitwise_rshift(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon64>) noexcept
        {
            return vshlq_s64(lhs, vnegq_s64(rhs));
        }

        /****************
         * bitwise_cast *
         ****************/

#define WRAP_CAST(SUFFIX, TYPE)                                                                                        \
    namespace wrap                                                                                                     \
    {                                                                                                                  \
        inline float64x2_t vreinterpretq_f64_##SUFFIX(TYPE a) noexcept { return ::vreinterpretq_f64_##SUFFIX(a); }     \
        inline TYPE vreinterpretq_##SUFFIX##_f64(float64x2_t a) noexcept { return ::vreinterpretq_##SUFFIX##_f64(a); } \
    }

        WRAP_CAST(u8, uint8x16_t)
        WRAP_CAST(s8, int8x16_t)
        WRAP_CAST(u16, uint16x8_t)
        WRAP_CAST(s16, int16x8_t)
        WRAP_CAST(u32, uint32x4_t)
        WRAP_CAST(s32, int32x4_t)
        WRAP_CAST(u64, uint64x2_t)
        WRAP_CAST(s64, int64x2_t)
        WRAP_CAST(f32, float32x4_t)

#undef WRAP_CAST

        template <class A, class T>
        inline batch<double, A> bitwise_cast(batch<T, A> const& arg, batch<double, A> const&, requires_arch<neon64>) noexcept
        {
            using caster_type = detail::bitwise_caster_impl<float64x2_t,
                                                            uint8x16_t, int8x16_t,
                                                            uint16x8_t, int16x8_t,
                                                            uint32x4_t, int32x4_t,
                                                            uint64x2_t, int64x2_t,
                                                            float32x4_t>;
            const caster_type caster = {
                std::make_tuple(wrap::vreinterpretq_f64_u8, wrap::vreinterpretq_f64_s8, wrap::vreinterpretq_f64_u16, wrap::vreinterpretq_f64_s16,
                                wrap::vreinterpretq_f64_u32, wrap::vreinterpretq_f64_s32, wrap::vreinterpretq_f64_u64, wrap::vreinterpretq_f64_s64,
                                wrap::vreinterpretq_f64_f32)
            };
            using register_type = typename batch<T, A>::register_type;
            return caster.apply(register_type(arg));
        }

        namespace detail
        {
            template <class S, class... R>
            struct bitwise_caster_neon64
            {
                using container_type = std::tuple<R (*)(S)...>;
                container_type m_func;

                template <class V>
                V apply(float64x2_t rhs) const
                {
                    using func_type = V (*)(float64x2_t);
                    auto func = xsimd::detail::get<func_type>(m_func);
                    return func(rhs);
                }
            };
        }

        template <class A, class R>
        inline batch<R, A> bitwise_cast(batch<double, A> const& arg, batch<R, A> const&, requires_arch<neon64>) noexcept
        {
            using caster_type = detail::bitwise_caster_neon64<float64x2_t,
                                                              uint8x16_t, int8x16_t,
                                                              uint16x8_t, int16x8_t,
                                                              uint32x4_t, int32x4_t,
                                                              uint64x2_t, int64x2_t,
                                                              float32x4_t>;
            const caster_type caster = {
                std::make_tuple(wrap::vreinterpretq_u8_f64, wrap::vreinterpretq_s8_f64, wrap::vreinterpretq_u16_f64, wrap::vreinterpretq_s16_f64,
                                wrap::vreinterpretq_u32_f64, wrap::vreinterpretq_s32_f64, wrap::vreinterpretq_u64_f64, wrap::vreinterpretq_s64_f64,
                                wrap::vreinterpretq_f32_f64)
            };
            using src_register_type = typename batch<double, A>::register_type;
            using dst_register_type = typename batch<R, A>::register_type;
            return caster.apply<dst_register_type>(src_register_type(arg));
        }

        template <class A>
        inline batch<double, A> bitwise_cast(batch<double, A> const& arg, batch<double, A> const&, requires_arch<neon64>) noexcept
        {
            return arg;
        }

        /*************
         * bool_cast *
         *************/

        template <class A>
        inline batch_bool<double, A> bool_cast(batch_bool<int64_t, A> const& arg, requires_arch<neon64>) noexcept
        {
            using register_type = typename batch_bool<int64_t, A>::register_type;
            return register_type(arg);
        }

        template <class A>
        inline batch_bool<int64_t, A> bool_cast(batch_bool<double, A> const& arg, requires_arch<neon64>) noexcept
        {
            using register_type = typename batch_bool<double, A>::register_type;
            return register_type(arg);
        }

        /*********
         * isnan *
         *********/

        template <class A>
        inline batch_bool<double, A> isnan(batch<double, A> const& arg, requires_arch<neon64>) noexcept
        {
            return !(arg == arg);
        }
    }

    template <class batch_type, typename batch_type::value_type... Values>
    struct batch_constant;

    namespace kernel
    {
        /***********
         * swizzle *
         ***********/

        namespace detail
        {
            using ::xsimd::batch_constant;
            using ::xsimd::detail::integer_sequence;
            using ::xsimd::detail::make_integer_sequence;

            template <class CB1, class CB2, class IS>
            struct index_burst_impl;

            template <class B1, class B2, typename B2::value_type... V,
                      typename B2::value_type... incr>
            struct index_burst_impl<batch_constant<B1>, batch_constant<B2, V...>,
                                    integer_sequence<typename B2::value_type, incr...>>
            {
                using type = batch_constant<B2, V...>;
            };

            template <class B1, typename B1::value_type V0, typename B1::value_type... V1,
                      class B2, typename B2::value_type... V2,
                      typename B2::value_type... incr>
            struct index_burst_impl<batch_constant<B1, V0, V1...>, batch_constant<B2, V2...>,
                                    integer_sequence<typename B2::value_type, incr...>>
            {
                using value_type = typename B2::value_type;
                using next_input = batch_constant<B1, V1...>;
                using next_output = batch_constant<B2, V2..., (V0 + incr)...>;
                using type = typename index_burst_impl<next_input, next_output, integer_sequence<value_type, incr...>>::type;
            };

            template <class B, class T>
            struct index_burst;

            template <class B, typename B::value_type... V, class T>
            struct index_burst<batch_constant<B, V...>, T>
            {
                static constexpr size_t mul = sizeof(typename B::value_type) / sizeof(T);
                using input = batch_constant<B, (mul * V)...>;
                using output = batch_constant<batch<T, typename B::arch_type>>;
                using type = typename index_burst_impl<input, output, make_integer_sequence<T, mul>>::type;
            };

            template <class B, class T>
            using index_burst_t = typename index_burst<B, T>::type;

            template <class T, class B>
            inline index_burst_t<B, T> burst_index(B)
            {
                return index_burst_t<B, T>();
            }
        }

        template <class A, uint8_t V0, uint8_t V1, uint8_t V2, uint8_t V3, uint8_t V4, uint8_t V5, uint8_t V6, uint8_t V7,
                  uint8_t V8, uint8_t V9, uint8_t V10, uint8_t V11, uint8_t V12, uint8_t V13, uint8_t V14, uint8_t V15>
        inline batch<uint8_t, A> swizzle(batch<uint8_t, A> const& self,
                                         batch_constant<batch<uint8_t, A>, V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15> idx,
                                         requires_arch<neon64>) noexcept
        {
            return vqtbl1q_u8(self, batch<uint8_t, A>(idx));
        }

        template <class A, uint8_t V0, uint8_t V1, uint8_t V2, uint8_t V3, uint8_t V4, uint8_t V5, uint8_t V6, uint8_t V7,
                  uint8_t V8, uint8_t V9, uint8_t V10, uint8_t V11, uint8_t V12, uint8_t V13, uint8_t V14, uint8_t V15>
        inline batch<int8_t, A> swizzle(batch<int8_t, A> const& self,
                                        batch_constant<batch<uint8_t, A>, V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15> idx,
                                        requires_arch<neon64>) noexcept
        {
            return vqtbl1q_s8(self, batch<uint8_t, A>(idx));
        }

        template <class A, uint16_t V0, uint16_t V1, uint16_t V2, uint16_t V3, uint16_t V4, uint16_t V5, uint16_t V6, uint16_t V7>
        inline batch<uint16_t, A> swizzle(batch<uint16_t, A> const& self,
                                          batch_constant<batch<uint16_t, A>, V0, V1, V2, V3, V4, V5, V6, V7> idx,
                                          requires_arch<neon64>) noexcept
        {
            using batch_type = batch<uint8_t, A>;
            return vreinterpretq_u16_u8(swizzle<A>(batch_type(vreinterpretq_u8_u16(self)), detail::burst_index<uint8_t>(idx), A()));
        }

        template <class A, uint16_t V0, uint16_t V1, uint16_t V2, uint16_t V3, uint16_t V4, uint16_t V5, uint16_t V6, uint16_t V7>
        inline batch<int16_t, A> swizzle(batch<int16_t, A> const& self,
                                         batch_constant<batch<uint16_t, A>, V0, V1, V2, V3, V4, V5, V6, V7> idx,
                                         requires_arch<neon64>) noexcept
        {
            using batch_type = batch<int8_t, A>;
            return vreinterpretq_s16_s8(swizzle<A>(batch_type(vreinterpretq_s8_s16(self)), detail::burst_index<uint8_t>(idx), A()));
        }

        template <class A, uint32_t V0, uint32_t V1, uint32_t V2, uint32_t V3>
        inline batch<uint32_t, A> swizzle(batch<uint32_t, A> const& self,
                                          batch_constant<batch<uint32_t, A>, V0, V1, V2, V3> idx,
                                          requires_arch<neon64>) noexcept
        {
            using batch_type = batch<uint8_t, A>;
            return vreinterpretq_u32_u8(swizzle<A>(batch_type(vreinterpretq_u8_u32(self)), detail::burst_index<uint8_t>(idx), A()));
        }

        template <class A, uint32_t V0, uint32_t V1, uint32_t V2, uint32_t V3>
        inline batch<int32_t, A> swizzle(batch<int32_t, A> const& self,
                                         batch_constant<batch<uint32_t, A>, V0, V1, V2, V3> idx,
                                         requires_arch<neon64>) noexcept
        {
            using batch_type = batch<int8_t, A>;
            return vreinterpretq_s32_s8(swizzle<A>(batch_type(vreinterpretq_s8_s32(self)), detail::burst_index<uint8_t>(idx), A()));
        }

        template <class A, uint64_t V0, uint64_t V1>
        inline batch<uint64_t, A> swizzle(batch<uint64_t, A> const& self,
                                          batch_constant<batch<uint64_t, A>, V0, V1> idx,
                                          requires_arch<neon64>) noexcept
        {
            using batch_type = batch<uint8_t, A>;
            return vreinterpretq_u64_u8(swizzle<A>(batch_type(vreinterpretq_u8_u64(self)), detail::burst_index<uint8_t>(idx), A()));
        }

        template <class A, uint64_t V0, uint64_t V1>
        inline batch<int64_t, A> swizzle(batch<int64_t, A> const& self,
                                         batch_constant<batch<uint64_t, A>, V0, V1> idx,
                                         requires_arch<neon64>) noexcept
        {
            using batch_type = batch<int8_t, A>;
            return vreinterpretq_s64_s8(swizzle<A>(batch_type(vreinterpretq_s8_s64(self)), detail::burst_index<uint8_t>(idx), A()));
        }

        template <class A, uint32_t V0, uint32_t V1, uint32_t V2, uint32_t V3>
        inline batch<float, A> swizzle(batch<float, A> const& self,
                                       batch_constant<batch<uint32_t, A>, V0, V1, V2, V3> idx,
                                       requires_arch<neon64>) noexcept
        {
            using batch_type = batch<uint8_t, A>;
            return vreinterpretq_f32_u8(swizzle<A>(batch_type(vreinterpretq_u8_f32(self)), detail::burst_index<uint8_t>(idx), A()));
        }

        template <class A, uint64_t V0, uint64_t V1>
        inline batch<double, A> swizzle(batch<double, A> const& self,
                                        batch_constant<batch<uint64_t, A>, V0, V1> idx,
                                        requires_arch<neon64>) noexcept
        {
            using batch_type = batch<uint8_t, A>;
            return vreinterpretq_f64_u8(swizzle<A>(batch_type(vreinterpretq_u8_f64(self)), detail::burst_index<uint8_t>(idx), A()));
        }
    }
}

#endif
