#ifndef XSIMD_NEON_HPP
#define XSIMD_NEON_HPP

#include <tuple>

#include "../types/xsimd_neon_register.hpp"
#include "../types/xsimd_utils.hpp"

namespace xsimd
{
    namespace kernel
    {
        using namespace types;

        namespace detail
        {
            template <template <class> class return_type, class... T>
            struct neon_dispatcher_base
            {
                struct unary
                {
                    using container_type = std::tuple<return_type<T> (*)(T)...>;
                    const container_type m_func;

                    template <class U>
                    return_type<U> apply(U rhs) const
                    {
                        using func_type = return_type<U> (*)(U);
                        auto func = xsimd::detail::get<func_type>(m_func);
                        return func(rhs);
                    }
                };

                struct binary
                {
                    using container_type = std::tuple<return_type<T> (*)(T, T) ...>;
                    const container_type m_func;

                    template <class U>
                    return_type<U> apply(U lhs, U rhs) const
                    {
                        using func_type = return_type<U> (*)(U, U);
                        auto func = xsimd::detail::get<func_type>(m_func);
                        return func(lhs, rhs);
                    }
                };
            };

            /***************************
             *  arithmetic dispatchers *
             ***************************/

            template <class T>
            using identity_return_type = T;
            
            template <class... T>
            struct neon_dispatcher_impl : neon_dispatcher_base<identity_return_type, T...>
            {
            };


            using neon_dispatcher = neon_dispatcher_impl<uint8x16_t, int8x16_t,
                                                       uint16x8_t, int16x8_t,
                                                       uint32x4_t, int32x4_t,
                                                       uint64x2_t, int64x2_t,
                                                       float32x4_t>;

            using excluding_int64_dispatcher = neon_dispatcher_impl<uint8x16_t, int8x16_t,
                                                                   uint16x8_t, int16x8_t,
                                                                   uint32x4_t, int32x4_t,
                                                                   float32x4_t>;

            /**************************
             * comparison dispatchers *
             **************************/

            template <class T>
            struct comp_return_type_impl;

            template <>
            struct comp_return_type_impl<uint8x16_t>
            {
                using type = uint8x16_t;
            };

            template <>
            struct comp_return_type_impl<int8x16_t>
            {
                using type = uint8x16_t;
            };

            template <>
            struct comp_return_type_impl<uint16x8_t>
            {
                using type = uint16x8_t;
            };

            template <>
            struct comp_return_type_impl<int16x8_t>
            {
                using type = uint16x8_t;
            };

            template <>
            struct comp_return_type_impl<uint32x4_t>
            {
                using type = uint32x4_t;
            };

            template <>
            struct comp_return_type_impl<int32x4_t>
            {
                using type = uint32x4_t;
            };

            template <>
            struct comp_return_type_impl<uint64x2_t>
            {
                using type = uint64x2_t;
            };

            template <>
            struct comp_return_type_impl<int64x2_t>
            {
                using type = uint64x2_t;
            };
            
            template <>
            struct comp_return_type_impl<float32x4_t>
            {
                using type = uint32x4_t;
            };

            template <class T>
            using comp_return_type = typename comp_return_type_impl<T>::type;

            template <class... T>
            struct neon_comp_dispatcher_impl : neon_dispatcher_base<comp_return_type, T...>
            {
            };

            using excluding_int64_comp_dispatcher = neon_comp_dispatcher_impl<uint8x16_t, int8x16_t,
                                                                             uint16x8_t, int16x8_t,
                                                                             uint32x4_t, int32x4_t,
                                                                             float32x4_t>;

            /**************************************
             * enabling / disabling metafunctions *
             **************************************/

            template <class T>
            using enable_integral_t = typename std::enable_if<std::is_integral<T>::value, int>::type;

            template <class T>
            using enable_neon_type_t = typename std::enable_if<std::is_integral<T>::value || std::is_same<T, float>::value,
                                                               int>::type;

            template <class T, size_t S>
            using enable_sized_signed_t = typename std::enable_if<std::is_integral<T>::value &&
                                                                  std::is_signed<T>::value &&
                                                                  sizeof(T) == S, int>::type;

            template <class T, size_t S>
            using enable_sized_unsigned_t = typename std::enable_if<std::is_integral<T>::value &&
                                                                    !std::is_signed<T>::value &&
                                                                    sizeof(T) == S, int>::type;

            template <class T, size_t S>
            using enable_sized_integral_t = typename std::enable_if<std::is_integral<T>::value &&
                                                                   sizeof(T) == S, int>::type;

            template <class T, size_t S>
            using enable_sized_t = typename std::enable_if<sizeof(T) == S, int>::type;

            template <class T>
            using exclude_int64_neon_t
                 = typename std::enable_if<(std::is_integral<T>::value && sizeof(T) != 8) || std::is_same<T, float>::value, int>::type;
        }

        /*************
         * broadcast *
         *************/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 1> = 0>
        batch<T, A> broadcast(T val, requires_arch<neon>)
        {
            return vdupq_n_u8(uint8_t(val));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 1> = 0>
        batch<T, A> broadcast(T val, requires_arch<neon>)
        {
            return vdupq_n_s8(int8_t(val));
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 2> = 0>
        batch<T, A> broadcast(T val, requires_arch<neon>)
        {
            return vdupq_n_u16(uint16_t(val));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 2> = 0>
        batch<T, A> broadcast(T val, requires_arch<neon>)
        {
            return vdupq_n_s16(int16_t(val));
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 4> = 0>
        batch<T, A> broadcast(T val, requires_arch<neon>)
        {
            return vdupq_n_u32(uint32_t(val));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 4> = 0>
        batch<T, A> broadcast(T val, requires_arch<neon>)
        {
            return vdupq_n_s32(int32_t(val));
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        batch<T, A> broadcast(T val, requires_arch<neon>)
        {
            return vdupq_n_u64(uint64_t(val));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        batch<T, A> broadcast(T val, requires_arch<neon>)
        {
            return vdupq_n_s64(int64_t(val));
        }

        template <class A>
        batch<float, A> broadcast(float val, requires_arch<neon>)
        {
            return vdupq_n_f32(val);
        }

        /*******
         * set *
         *******/

        template <class A, class T, class... Args, detail::enable_integral_t<T> = 0>
        batch<T, A> set(batch<T, A> const&, requires_arch<neon>, Args... args)
        {
            return xsimd::types::detail::neon_vector_type<T>{args...};
        }

        template <class A, class T, class... Args, detail::enable_integral_t<T> = 0>
        batch_bool<T, A> set(batch_bool<T, A> const&, requires_arch<neon>, Args... args)
        {
            using register_type = typename batch_bool<T, A>::register_type;
            using unsigned_type = as_unsigned_integer_t<T>;
            return register_type{static_cast<unsigned_type>(args ? -1LL : 0LL)...};
        }

        template <class A>
        batch<float, A> set(batch<float, A> const&, requires_arch<neon>, float f0, float f1, float f2, float f3)
        {
            return float32x4_t{f0, f1, f2, f3};
        }

        template <class A, class... Args>
        batch_bool<float, A> set(batch_bool<float, A> const&, requires_arch<neon>, Args... args)
        {
            using register_type = typename batch_bool<float, A>::register_type;
            using unsigned_type = as_unsigned_integer_t<float>;
            return register_type{static_cast<unsigned_type>(args ? -1LL : 0LL)...};
        }

        /*************
         * from_bool *
         *************/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 1> = 0>
        batch<T, A> from_bool(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            return vandq_u8(arg, vdupq_n_u8(1));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 1> = 0>
        batch<T, A> from_bool(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            return vandq_s8(reinterpret_cast<int8x16_t>(arg.data), vdupq_n_s8(1));
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 2> = 0>
        batch<T, A> from_bool(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            return vandq_u16(arg, vdupq_n_u16(1));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 2> = 0>
        batch<T, A> from_bool(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            return vandq_s16(reinterpret_cast<int16x8_t>(arg.data), vdupq_n_s16(1));
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 4> = 0>
        batch<T, A> from_bool(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            return vandq_u32(arg, vdupq_n_u32(1));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 4> = 0>
        batch<T, A> from_bool(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            return vandq_s32(reinterpret_cast<int32x4_t>(arg.data), vdupq_n_s32(1));
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        batch<T, A> from_bool(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            return vandq_u64(arg, vdupq_n_u64(1));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        batch<T, A> from_bool(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            return vandq_s64(reinterpret_cast<int64x2_t>(arg.data), vdupq_n_s64(1));
        }

        template <class A>
        batch<float, A> from_bool(batch_bool<float, A> const& arg, requires_arch<neon>)
        {
            return vreinterpretq_f32_u32(vandq_u32(arg, vreinterpretq_u32_f32(vdupq_n_f32(1.f))));
        }

        /********
         * load *
         ********/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 1> = 0>
        batch<T, A> load_aligned(T const* src, convert<T>, requires_arch<neon>)
        {
            return vld1q_u8(src);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 1> = 0>
        batch<T, A> load_aligned(T const* src, convert<T>, requires_arch<neon>)
        {
            return vld1q_s8(src);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 2> = 0>
        batch<T, A> load_aligned(T const* src, convert<T>, requires_arch<neon>)
        {
            return vld1q_u16(src);
        }
        template <class A, class T, detail::enable_sized_signed_t<T, 2> = 0>
        batch<T, A> load_aligned(T const* src, convert<T>, requires_arch<neon>)
        {
            return vld1q_s16(src);
        }
        template <class A, class T, detail::enable_sized_unsigned_t<T, 4> = 0>
        batch<T, A> load_aligned(T const* src, convert<T>, requires_arch<neon>)
        {
            return vld1q_u32(src);
        }
        template <class A, class T, detail::enable_sized_signed_t<T, 4> = 0>
        batch<T, A> load_aligned(T const* src, convert<T>, requires_arch<neon>)
        {
            return vld1q_s32(src);
        }
        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        batch<T, A> load_aligned(T const* src, convert<T>, requires_arch<neon>)
        {
            return vld1q_u64(src);
        }
        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        batch<T, A> load_aligned(T const* src, convert<T>, requires_arch<neon>)
        {
            return vld1q_s64(src);
        }

        template <class A>
        batch<float, A> load_aligned(float const* src, convert<float>, requires_arch<neon>)
        {
            return vld1q_f32(src);
        }

        template <class A, class T>
        batch<T, A> load_unaligned(T const* src, convert<T>, requires_arch<neon>)
        {
            return load_aligned<A>(src, convert<T>(), A{});
        }

        /*********
         * store *
         *********/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 1> = 0>
        void store_aligned(T* dst, batch<T, A> const& src, requires_arch<neon>)
        {
            vst1q_u8(dst, src);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 1> = 0>
        void store_aligned(T* dst, batch<T, A> const& src, requires_arch<neon>)
        {
            vst1q_s8(dst, src);
        }
        
        template <class A, class T, detail::enable_sized_unsigned_t<T, 2> = 0>
        void store_aligned(T* dst, batch<T, A> const& src, requires_arch<neon>)
        {
            vst1q_u16(dst, src);
        }
        
        template <class A, class T, detail::enable_sized_signed_t<T, 2> = 0>
        void store_aligned(T* dst, batch<T, A> const& src, requires_arch<neon>)
        {
            vst1q_s16(dst, src);
        }
        
        template <class A, class T, detail::enable_sized_unsigned_t<T, 4> = 0>
        void store_aligned(T* dst, batch<T, A> const& src, requires_arch<neon>)
        {
            vst1q_u32(dst, src);
        }
        
        template <class A, class T, detail::enable_sized_signed_t<T, 4> = 0>
        void store_aligned(T* dst, batch<T, A> const& src, requires_arch<neon>)
        {
            vst1q_s32(dst, src);
        }
        
        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        void store_aligned(T* dst, batch<T, A> const& src, requires_arch<neon>)
        {
            vst1q_u64(dst, src);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        void store_aligned(T* dst, batch<T, A> const& src, requires_arch<neon>)
        {
            vst1q_s64(dst, src);
        }

        template <class A>
        void store_aligned(float* dst, batch<float, A> const& src, requires_arch<neon>)
        {
            vst1q_f32(dst, src);
        }

        template <class A, class T>
        void store_unaligned(T* dst, batch<T, A> const& src, requires_arch<neon>)
        {
            store_aligned<A>(dst, src, A{});
        }

        /****************
         * load_complex *
         ****************/

        template <class A>
        batch<std::complex<float>, A> load_complex_aligned(std::complex<float> const* mem, requires_arch<neon>)
        {
            using real_batch = batch<float, A>;
            const float* buf = reinterpret_cast<const float*>(mem);
            float32x4x2_t tmp = vld2q_f32(buf);
            real_batch real = tmp.val[0],
                       imag = tmp.val[1];
            return batch<std::complex<float>, A>{real, imag};
        }

        template <class A>
        batch<std::complex<float>, A> load_complex_unaligned(std::complex<float> const* mem, requires_arch<neon>)
        {
            return load_complex_aligned<A>(mem, A{});
        }

        /*****************
         * store_complex *
         *****************/

        template <class A>
        void store_complex_aligned(std::complex<float>* dst, batch<std::complex<float> ,A> const& src, requires_arch<neon>)
        {
            float32x4x2_t tmp;
            tmp.val[0] = src.real();
            tmp.val[1] = src.imag();
            float* buf = reinterpret_cast<float*>(dst);
            vst2q_f32(buf, tmp);
        }

        template <class A>
        void store_complex_unaligned(std::complex<float>* dst, batch<std::complex<float> ,A> const& src, requires_arch<neon>)
        {
            store_complex_aligned(dst, src, A{});
        }

        /*******
         * neg *
         *******/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 1> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vreinterpretq_u8_s8(vnegq_s8(vreinterpretq_s8_u8(rhs)));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 1> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vnegq_s8(rhs);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 2> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vreinterpretq_u16_s16(vnegq_s16(vreinterpretq_s16_u16(rhs)));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 2> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vnegq_s16(rhs);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 4> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vreinterpretq_u32_s32(vnegq_s32(vreinterpretq_s32_u32(rhs)));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 4> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vnegq_s32(rhs);
        }

        template <class A, class T,  detail::enable_sized_unsigned_t<T, 8> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires_arch<neon>)
        {
            return batch<T, A>({-rhs.get(0), -rhs.get(1)});
        }

        template <class A, class T,  detail::enable_sized_signed_t<T, 8> = 0>
        batch<T, A> neg(batch<T, A> const& rhs, requires_arch<neon>)
        {
            return batch<T, A>({-rhs.get(0), -rhs.get(1)});
        }
        
        template <class A>
        batch<float, A> neg(batch<float, A> const& rhs, requires_arch<neon>)
        {
            return vnegq_f32(rhs);
        }

        /*******
         * add *
         *******/

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch<T, A> add(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::neon_dispatcher::binary dispatcher =
            {
                std::make_tuple(vaddq_u8, vaddq_s8, vaddq_u16, vaddq_s16,
                                vaddq_u32, vaddq_s32, vaddq_u64, vaddq_s64,
                                vaddq_f32)
            };
            return dispatcher.apply(register_type(lhs), register_type(rhs));
        }

        /********
         * sadd *
         ********/

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch<T, A> sadd(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::neon_dispatcher::binary dispatcher =
            {
                std::make_tuple(vqaddq_u8, vqaddq_s8, vqaddq_u16, vqaddq_s16,
                                vqaddq_u32, vqaddq_s32, vqaddq_u64, vqaddq_s64,
                                vaddq_f32)
            };
            return dispatcher.apply(register_type(lhs), register_type(rhs));
        }

        /*******
         * sub *
         *******/

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch<T, A> sub(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::neon_dispatcher::binary dispatcher =
            {
                std::make_tuple(vsubq_u8, vsubq_s8, vsubq_u16, vsubq_s16,
                                vsubq_u32, vsubq_s32, vsubq_u64, vsubq_s64,
                                vsubq_f32)
            };
            return dispatcher.apply(register_type(lhs), register_type(rhs));
        }

        /********
         * ssub *
         ********/

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch<T, A> ssub(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::neon_dispatcher::binary dispatcher =
            {
                std::make_tuple(vqsubq_u8, vqsubq_s8, vqsubq_u16, vqsubq_s16,
                                vqsubq_u32, vqsubq_s32, vqsubq_u64, vqsubq_s64,
                                vsubq_f32)
            };
            return dispatcher.apply(register_type(lhs), register_type(rhs));
        }


        /*******
         * mul *
         *******/

        template <class A, class T, detail::exclude_int64_neon_t<T> = 0>
        batch<T, A> mul(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_dispatcher::binary dispatcher =
            {
                std::make_tuple(vmulq_u8, vmulq_s8, vmulq_u16, vmulq_s16, vmulq_u32, vmulq_s32, vmulq_f32)
            };
            return dispatcher.apply(register_type(lhs), register_type(rhs));
        }

        /*******
         * div *
         *******/

#if defined(XSIMD_FAST_INTEGER_DIVISION)
        template <class A, class T,  detail::enable_sized_signed_t<T, 4> = 0>
        batch<T, A> div(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vcvtq_s32_f32(vcvtq_f32_s32(lhs) / vcvtq_f32_s32(rhs));
        }

        template <class A, class T,  detail::enable_sized_unsigned_t<T, 4> = 0>
        batch<T, A> div(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vcvtq_u32_f32(vcvtq_f32_u32(lhs) / vcvtq_f32_u32(rhs));
        }
#endif

        template <class A>
        batch<float, A> div(batch<float, A> const& lhs, batch<float, A> const& rhs, requires_arch<neon>)
        {
            // from stackoverflow & https://projectne10.github.io/Ne10/doc/NE10__divc_8neon_8c_source.html
            // get an initial estimate of 1/b.
            float32x4_t reciprocal = vrecpeq_f32(rhs);

            // use a couple Newton-Raphson steps to refine the estimate.  Depending on your
            // application's accuracy requirements, you may be able to get away with only
            // one refinement (instead of the two used here).  Be sure to test!
            reciprocal = vmulq_f32(vrecpsq_f32(rhs, reciprocal), reciprocal);
            reciprocal = vmulq_f32(vrecpsq_f32(rhs, reciprocal), reciprocal);

            // and finally, compute a / b = a * (1 / b)
            return vmulq_f32(lhs, reciprocal);
        }

        /******
         * eq *
         ******/

        template <class A, class T, detail::exclude_int64_neon_t<T> = 0>
        batch_bool<T, A> eq(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_comp_dispatcher::binary dispatcher =
            {
                std::make_tuple(vceqq_u8, vceqq_s8, vceqq_u16, vceqq_s16, vceqq_u32, vceqq_s32, vceqq_f32)
            };
            return dispatcher.apply(register_type(lhs), register_type(rhs));
        }

        template <class A, class T, detail::exclude_int64_neon_t<T> = 0>
        batch_bool<T, A> eq(batch_bool<T, A> const& lhs, batch_bool<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch_bool<T, A>::register_type;
            using dispatcher_type = detail::neon_comp_dispatcher_impl<uint8x16_t, uint16x8_t, uint32x4_t>::binary;
            constexpr dispatcher_type dispatcher =
            {
                std::make_tuple(vceqq_u8, vceqq_u16, vceqq_u32)
            };
            return dispatcher.apply(register_type(lhs), register_type(rhs));
        }

        template <class A, class T, detail::enable_sized_integral_t<T, 8> = 0>
        batch_bool<T, A> eq(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return batch_bool<T, A>({lhs.get(0) == rhs.get(0), lhs.get(1) == rhs.get(1)});
        }

        template <class A, class T, detail::enable_sized_integral_t<T, 8> = 0>
        batch_bool<T, A> eq(batch_bool<T, A> const& lhs, batch_bool<T, A> const& rhs, requires_arch<neon>)
        {
            return batch_bool<T, A>({lhs.get(0) == rhs.get(0), lhs.get(1) == rhs.get(1)});
        }

        /******
         * lt *
         ******/

        template <class A, class T, detail::exclude_int64_neon_t<T> = 0>
        batch_bool<T, A> lt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_comp_dispatcher::binary dispatcher =
            {
                std::make_tuple(vcltq_u8, vcltq_s8, vcltq_u16, vcltq_s16, vcltq_u32, vcltq_s32, vcltq_f32)
            };
            return dispatcher.apply(register_type(lhs), register_type(rhs));
        }

        template <class A, class T, detail::enable_sized_integral_t<T, 8> = 0>
        batch_bool<T, A> lt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return batch_bool<T, A>({lhs.get(0) < rhs.get(0), lhs.get(1) < rhs.get(1)});
        }

        /******
         * le *
         ******/

        template <class A, class T, detail::exclude_int64_neon_t<T> = 0>
        batch_bool<T, A> le(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_comp_dispatcher::binary dispatcher =
            {
                std::make_tuple(vcleq_u8, vcleq_s8, vcleq_u16, vcleq_s16, vcleq_u32, vcleq_s32, vcleq_f32)
            };
            return dispatcher.apply(register_type(lhs), register_type(rhs));
        }

        template <class A, class T, detail::enable_sized_integral_t<T, 8> = 0>
        batch_bool<T, A> le(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return batch_bool<T, A>({lhs.get(0) <= rhs.get(0), lhs.get(1) <= rhs.get(1)});
        }

        /******
         * gt *
         ******/

        template <class A, class T, detail::exclude_int64_neon_t<T> = 0>
        batch_bool<T, A> gt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_comp_dispatcher::binary dispatcher =
            {
                std::make_tuple(vcgtq_u8, vcgtq_s8, vcgtq_u16, vcgtq_s16, vcgtq_u32, vcgtq_s32, vcgtq_f32)
            };
            return dispatcher.apply(register_type(lhs), register_type(rhs));
        }

        template <class A, class T, detail::enable_sized_integral_t<T, 8> = 0>
        batch_bool<T, A> gt(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return batch_bool<T, A>({lhs.get(0) > rhs.get(0), lhs.get(1) > rhs.get(1)});
        }

        /******
         * ge *
         ******/

        template <class A, class T, detail::exclude_int64_neon_t<T> = 0>
        batch_bool<T, A> ge(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_comp_dispatcher::binary dispatcher =
            {
                std::make_tuple(vcgeq_u8, vcgeq_s8, vcgeq_u16, vcgeq_s16, vcgeq_u32, vcgeq_s32, vcgeq_f32)
            };
            return dispatcher.apply(register_type(lhs), register_type(rhs));
        }

        template <class A, class T, detail::enable_sized_integral_t<T, 8> = 0>
        batch_bool<T, A> ge(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return batch_bool<T, A>({lhs.get(0) >= rhs.get(0), lhs.get(1) >= rhs.get(1)});
        }

        /***************
         * bitwise_and *
         ***************/

        namespace detail
        {
            inline float32x4_t bitwise_and_f32(float32x4_t lhs, float32x4_t rhs)
            {
                return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(lhs),
                                                       vreinterpretq_u32_f32(rhs)));
            }

            template <class V>
            V bitwise_and_neon(V const& lhs, V const& rhs)
            {
                constexpr neon_dispatcher::binary dispatcher =
                {
                    std::make_tuple(vandq_u8, vandq_s8, vandq_u16, vandq_s16,
                                    vandq_u32, vandq_s32, vandq_u64, vandq_s64,
                                    bitwise_and_f32)
                };
                return dispatcher.apply(lhs, rhs);
            }
        }

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch<T, A> bitwise_and(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            return detail::bitwise_and_neon(register_type(lhs), register_type(rhs));
        }

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch_bool<T, A> bitwise_and(batch_bool<T, A> const& lhs, batch_bool<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch_bool<T, A>::register_type;
            return detail::bitwise_and_neon(register_type(lhs), register_type(rhs));
        }

        /**************
         * bitwise_or *
         **************/

        namespace detail
        {
            inline float32x4_t bitwise_or_f32(float32x4_t lhs, float32x4_t rhs)
            {
                return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(lhs),
                                                       vreinterpretq_u32_f32(rhs)));
            }

            template <class V>
            V bitwise_or_neon(V const& lhs, V const& rhs)
            {
                constexpr neon_dispatcher::binary dispatcher =
                {
                    std::make_tuple(vorrq_u8, vorrq_s8, vorrq_u16, vorrq_s16,
                                    vorrq_u32, vorrq_s32, vorrq_u64, vorrq_s64,
                                    bitwise_or_f32)
                };
                return dispatcher.apply(lhs, rhs);
            }
        }

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch<T, A> bitwise_or(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            return detail::bitwise_or_neon(register_type(lhs), register_type(rhs));
        }

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch_bool<T, A> bitwise_or(batch_bool<T, A> const& lhs, batch_bool<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch_bool<T, A>::register_type;
            return detail::bitwise_or_neon(register_type(lhs), register_type(rhs));
        }

        /***************
         * bitwise_xor *
         ***************/

        namespace detail
        {
            inline float32x4_t bitwise_xor_f32(float32x4_t lhs, float32x4_t rhs)
            {
                return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(lhs),
                                                       vreinterpretq_u32_f32(rhs)));
            }

            template <class V>
            V bitwise_xor_neon(V const& lhs, V const& rhs)
            {
                constexpr neon_dispatcher::binary dispatcher =
                {
                    std::make_tuple(veorq_u8, veorq_s8, veorq_u16, veorq_s16,
                                    veorq_u32, veorq_s32, veorq_u64, veorq_s64,
                                    bitwise_xor_f32)
                };
                return dispatcher.apply(lhs, rhs);
            }
        }

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch<T, A> bitwise_xor(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            return detail::bitwise_xor_neon(register_type(lhs), register_type(rhs));
        }

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch_bool<T, A> bitwise_xor(batch_bool<T, A> const& lhs, batch_bool<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch_bool<T, A>::register_type;
            return detail::bitwise_xor_neon(register_type(lhs), register_type(rhs));
        }

        /*******
         * neq *
         *******/

        template <class A, class T>
        batch_bool<T, A> neq(batch_bool<T, A> const& lhs, batch_bool<T, A> const& rhs, requires_arch<neon>)
        {
            return bitwise_xor(lhs, rhs, A{});
        }

        /***************
         * bitwise_not *
         ***************/

        namespace detail
        {
            inline int64x2_t bitwise_not_s64(int64x2_t arg)
            {
                return vreinterpretq_s64_s32(vmvnq_s32(vreinterpretq_s32_s64(arg)));
            }

            inline uint64x2_t bitwise_not_u64(uint64x2_t arg)
            {
                return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(arg)));
            }

            inline float32x4_t bitwise_not_f32(float32x4_t arg)
            {
                return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(arg)));
            }

            template <class V>
            V bitwise_not_neon(V const& arg)
            {
                constexpr neon_dispatcher::unary dispatcher =
                {
                    std::make_tuple(vmvnq_u8, vmvnq_s8, vmvnq_u16, vmvnq_s16,
                                    vmvnq_u32, vmvnq_s32,
                                    bitwise_not_u64, bitwise_not_s64,
                                    bitwise_not_f32)
                };
                return dispatcher.apply(arg);
            }
        }

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch<T, A> bitwise_not(batch<T, A> const& arg, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            return detail::bitwise_not_neon(register_type(arg));
        }

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch_bool<T, A> bitwise_not(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            using register_type = typename batch_bool<T, A>::register_type;
            return detail::bitwise_not_neon(register_type(arg));
        }

        /******************
         * bitwise_andnot *
         ******************/

        namespace detail
        {
            inline float32x4_t bitwise_andnot_f32(float32x4_t lhs, float32x4_t rhs)
            {
                return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
            }

            template <class V>
            V bitwise_andnot_neon(V const& lhs, V const& rhs)
            {
                constexpr detail::neon_dispatcher::binary dispatcher =
                {
                    std::make_tuple(vbicq_u8, vbicq_s8, vbicq_u16, vbicq_s16,
                                    vbicq_u32, vbicq_s32, vbicq_u64, vbicq_s64,
                                    bitwise_andnot_f32)
                };
                return dispatcher.apply(lhs, rhs);
            }
        }

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch<T, A> bitwise_andnot(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            return detail::bitwise_andnot_neon(register_type(lhs), register_type(rhs));
        }

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch_bool<T, A> bitwise_andnot(batch_bool<T, A> const& lhs, batch_bool<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch_bool<T, A>::register_type;
            return detail::bitwise_andnot_neon(register_type(lhs), register_type(rhs));
        }

        /*******
         * min *
         *******/

        template <class A, class T, detail::exclude_int64_neon_t<T> = 0>
        batch<T, A> min(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_dispatcher::binary dispatcher = 
            {
                std::make_tuple(vminq_u8, vminq_s8, vminq_u16, vminq_s16,
                                vminq_u32, vminq_s32, vminq_f32)
            };
            return dispatcher.apply(register_type(lhs), register_type(rhs));
        }

        template <class A, class T, detail::enable_sized_integral_t<T, 8> = 0>
        batch<T, A> min(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return { std::min(lhs.get(0), rhs.get(0)), std::min(lhs.get(1), rhs.get(1)) };
        }

        /*******
         * max *
         *******/

        template <class A, class T, detail::exclude_int64_neon_t<T> = 0>
        batch<T, A> max(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_dispatcher::binary dispatcher = 
            {
                std::make_tuple(vmaxq_u8, vmaxq_s8, vmaxq_u16, vmaxq_s16,
                                vmaxq_u32, vmaxq_s32, vmaxq_f32)
            };
            return dispatcher.apply(register_type(lhs), register_type(rhs));
        }

        template <class A, class T, detail::enable_sized_integral_t<T, 8> = 0>
        batch<T, A> max(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return { std::max(lhs.get(0), rhs.get(0)), std::max(lhs.get(1), rhs.get(1)) };
        }

        /*******
         * abs *
         *******/

        namespace detail
        {
            inline uint8x16_t abs_u8(uint8x16_t arg)
            {
                return arg;
            }

            inline uint16x8_t abs_u16(uint16x8_t arg)
            {
                return arg;
            }

            inline uint32x4_t abs_u32(uint32x4_t arg)
            {
                return arg;
            }
        }

        template <class A, class T, detail::exclude_int64_neon_t<T> = 0>
        batch<T, A> abs(batch<T, A> const& arg, requires_arch<neon>)
        {
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::excluding_int64_dispatcher::unary dispatcher = 
            {
                std::make_tuple(detail::abs_u8, vabsq_s8, detail::abs_u16, vabsq_s16,
                                detail::abs_u32, vabsq_s32, vabsq_f32)
            };
            return dispatcher.apply(register_type(arg));
        }

        /********
         * sqrt *
         ********/

        template <class A>
        batch<float, A> sqrt(batch<float, A> const& arg, requires_arch<neon>)
        {
            batch<float, A> sqrt_reciprocal = vrsqrteq_f32(arg);
            // one iter
            sqrt_reciprocal = sqrt_reciprocal * batch<float, A>(vrsqrtsq_f32(arg * sqrt_reciprocal, sqrt_reciprocal));
            batch<float, A> sqrt_approx = arg * sqrt_reciprocal * batch<float, A>(vrsqrtsq_f32(arg * sqrt_reciprocal, sqrt_reciprocal));
            batch<float, A> zero(0.f);
            return select(arg == zero, zero, sqrt_approx);
        }

        /********************
         * Fused operations *
         ********************/

#ifdef __ARM_FEATURE_FMA
        template <class A>
        batch<float, A> fma(batch<float, A> const& x, batch<float, A> const& y, batch<float, A> const& z, requires_arch<neon>)
        {
            return vfmaq_f32(z, x, y);
        }

        template <class A>
        batch<float, A> fms(batch<float, A> const& x, batch<float, A> const& y, batch<float, A> const& z, requires_arch<neon>)
        {
            return vfmaq_f32(-z, x, y);
        }
#endif

        /********
         * hadd *
         ********/

        namespace detail
        {
            template <class T, class A, class V>
            T sum_batch(V const& arg)
            {
                T res = T(0);
                for (std::size_t i = 0; i < batch<T, A>::size; ++i)
                {
                    res += arg[i];
                }
                return res;
            }
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 1> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon>)
        {
            uint8x8_t tmp = vpadd_u8(vget_low_u8(arg), vget_high_u8(arg));
            return detail::sum_batch<T, A>(tmp);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 1> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon>)
        {
            int8x8_t tmp = vpadd_s8(vget_low_s8(arg), vget_high_s8(arg));
            return detail::sum_batch<T, A>(tmp);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 2> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon>)
        {
            uint16x4_t tmp = vpadd_u16(vget_low_u16(arg), vget_high_u16(arg));
            return detail::sum_batch<T, A>(tmp);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 2> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon>)
        {
            int16x4_t tmp = vpadd_s16(vget_low_s16(arg), vget_high_s16(arg));
            return detail::sum_batch<T, A>(tmp);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 4> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon>)
        {
            uint32x2_t tmp = vpadd_u32(vget_low_u32(arg), vget_high_u32(arg));
            tmp = vpadd_u32(tmp, tmp);
            return vget_lane_u32(tmp, 0);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 4> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon>)
        {
            int32x2_t tmp = vpadd_s32(vget_low_s32(arg), vget_high_s32(arg));
            tmp = vpadd_s32(tmp, tmp);
            return vget_lane_s32(tmp, 0);
        }

        template <class A, class T, detail::enable_sized_integral_t<T, 8> = 0>
        typename batch<T, A>::value_type hadd(batch<T, A> const& arg, requires_arch<neon>)
        {
            return arg.get(0) + arg.get(1);
        }

        template <class A>
        float hadd(batch<float, A> const& arg, requires_arch<neon>)
        {
            float32x2_t tmp = vpadd_f32(vget_low_f32(arg), vget_high_f32(arg));
            tmp = vpadd_f32(tmp, tmp);
            return vget_lane_f32(tmp, 0);
        }

        /*********
         * haddp *
         *********/

        template <class A>
        batch<float, A> haddp(const batch<float, A>* row, requires_arch<neon>)
        {
            // row = (a,b,c,d)
            float32x2_t tmp1, tmp2, tmp3;
            // tmp1 = (a0 + a2, a1 + a3)
            tmp1 = vpadd_f32(vget_low_f32(row[0]), vget_high_f32(row[0]));
            // tmp2 = (b0 + b2, b1 + b3)
            tmp2 = vpadd_f32(vget_low_f32(row[1]), vget_high_f32(row[1]));
            // tmp1 = (a0..3, b0..3)
            tmp1 = vpadd_f32(tmp1, tmp2);
            // tmp2 = (c0 + c2, c1 + c3)
            tmp2 = vpadd_f32(vget_low_f32(row[2]), vget_high_f32(row[2]));
            // tmp3 = (d0 + d2, d1 + d3)
            tmp3 = vpadd_f32(vget_low_f32(row[3]), vget_high_f32(row[3]));
            // tmp1 = (c0..3, d0..3)
            tmp2 = vpadd_f32(tmp2, tmp3);
            // return = (a0..3, b0..3, c0..3, d0..3)
            return vcombine_f32(tmp1, tmp2);
        }

        /**********
         * select *
         **********/

        namespace detail
        {
            template <class... T>
            struct neon_select_dispatcher_impl
            {
                using container_type = std::tuple<T (*)(comp_return_type<T>, T, T)...>;
                const container_type m_func;

                template <class U>
                U apply(comp_return_type<U> cond, U lhs, U rhs) const
                {
                    using func_type = U (*)(comp_return_type<U>, U, U);
                    auto func = xsimd::detail::get<func_type>(m_func);
                    return func(cond, lhs, rhs);
                }
            };

            using neon_select_dispatcher = neon_select_dispatcher_impl<uint8x16_t, int8x16_t,
                                                                     uint16x8_t, int16x8_t,
                                                                     uint32x4_t, int32x4_t,
                                                                     uint64x2_t, int64x2_t,
                                                                     float32x4_t>;
        }

        template <class A, class T, detail::enable_neon_type_t<T> = 0>
        batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& a, batch<T, A> const& b, requires_arch<neon>)
        {
            using bool_register_type = typename batch_bool<T, A>::register_type;
            using register_type = typename batch<T, A>::register_type;
            constexpr detail::neon_select_dispatcher dispatcher =
            {
                std::make_tuple(vbslq_u8, vbslq_s8, vbslq_u16, vbslq_s16,
                                vbslq_u32, vbslq_s32, vbslq_u64, vbslq_s64,
                                vbslq_f32)
            };
            return dispatcher.apply(bool_register_type(cond), register_type(a), register_type(b));
        }

        template <class A, class T, bool... b, detail::enable_neon_type_t<T> = 0>
        batch<T, A> select(batch_bool_constant<batch<T, A>, b...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<neon>)
        {
            return select(batch_bool<T, A>{b...}, true_br, false_br, neon{});
        }

        /**********
         * zip_lo *
         **********/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 1> = 0>
        batch<T, A> zip_lo(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            uint8x8x2_t tmp = vzip_u8(vget_low_u8(lhs), vget_low_u8(rhs));
            return vcombine_u8(tmp.val[0], tmp.val[1]);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 1> = 0>
        batch<T, A> zip_lo(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            int8x8x2_t tmp = vzip_s8(vget_low_s8(lhs), vget_low_s8(rhs));
            return vcombine_s8(tmp.val[0], tmp.val[1]);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 2> = 0>
        batch<T, A> zip_lo(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            uint16x4x2_t tmp = vzip_u16(vget_low_u16(lhs), vget_low_u16(rhs));
            return vcombine_u16(tmp.val[0], tmp.val[1]);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 2> = 0>
        batch<T, A> zip_lo(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            int16x4x2_t tmp = vzip_s16(vget_low_s16(lhs), vget_low_s16(rhs));
            return vcombine_s16(tmp.val[0], tmp.val[1]);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 4> = 0>
        batch<T, A> zip_lo(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            uint32x2x2_t tmp = vzip_u32(vget_low_u32(lhs), vget_low_u32(rhs));
            return vcombine_u32(tmp.val[0], tmp.val[1]);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 4> = 0>
        batch<T, A> zip_lo(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            int32x2x2_t tmp = vzip_s32(vget_low_s32(lhs), vget_low_s32(rhs));
            return vcombine_s32(tmp.val[0], tmp.val[1]);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        batch<T, A> zip_lo(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vcombine_u64(vget_low_u64(lhs), vget_low_u64(rhs));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        batch<T, A> zip_lo(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vcombine_s64(vget_low_s64(lhs), vget_low_s64(rhs));
        }

        template <class A>
        batch<float, A> zip_lo(batch<float, A> const& lhs, batch<float, A> const& rhs, requires_arch<neon>)
        {
            float32x2x2_t tmp = vzip_f32(vget_low_f32(lhs), vget_low_f32(rhs));
            return vcombine_f32(tmp.val[0], tmp.val[1]);
        }

        /**********
         * zip_hi *
         **********/

        template <class A, class T, detail::enable_sized_unsigned_t<T, 1> = 0>
        batch<T, A> zip_hi(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            uint8x8x2_t tmp = vzip_u8(vget_high_u8(lhs), vget_high_u8(rhs));
            return vcombine_u8(tmp.val[0], tmp.val[1]);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 1> = 0>
        batch<T, A> zip_hi(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            int8x8x2_t tmp = vzip_s8(vget_high_s8(lhs), vget_high_s8(rhs));
            return vcombine_s8(tmp.val[0], tmp.val[1]);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 2> = 0>
        batch<T, A> zip_hi(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            uint16x4x2_t tmp = vzip_u16(vget_high_u16(lhs), vget_high_u16(rhs));
            return vcombine_u16(tmp.val[0], tmp.val[1]);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 2> = 0>
        batch<T, A> zip_hi(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            int16x4x2_t tmp = vzip_s16(vget_high_s16(lhs), vget_high_s16(rhs));
            return vcombine_s16(tmp.val[0], tmp.val[1]);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 4> = 0>
        batch<T, A> zip_hi(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            uint32x2x2_t tmp = vzip_u32(vget_high_u32(lhs), vget_high_u32(rhs));
            return vcombine_u32(tmp.val[0], tmp.val[1]);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 4> = 0>
        batch<T, A> zip_hi(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            int32x2x2_t tmp = vzip_s32(vget_high_s32(lhs), vget_high_s32(rhs));
            return vcombine_s32(tmp.val[0], tmp.val[1]);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        batch<T, A> zip_hi(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vcombine_u64(vget_high_u64(lhs), vget_high_u64(rhs));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        batch<T, A> zip_hi(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vcombine_s64(vget_high_s64(lhs), vget_high_s64(rhs));
        }

        template <class A>
        batch<float, A> zip_hi(batch<float, A> const& lhs, batch<float, A> const& rhs, requires_arch<neon>)
        {
            float32x2x2_t tmp = vzip_f32(vget_high_f32(lhs), vget_high_f32(rhs));
            return vcombine_f32(tmp.val[0], tmp.val[1]);
        }

        /****************
         * extract_pair *
         ****************/

        namespace detail
        {
            template <class A, class T>
            batch<T, A> extract_pair(batch<T, A> const&, batch<T, A> const& /*rhs*/, std::size_t, ::xsimd::detail::index_sequence<>)
            {
                assert(false && "extract_pair out of bounds");
                return  batch<T, A>{};
            }

            template <class A, class T, size_t I, size_t... Is, detail::enable_sized_unsigned_t<T, 1> = 0>
            batch<T, A> extract_pair(batch<T, A> const& lhs, batch<T, A> const& rhs, std::size_t n, ::xsimd::detail::index_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vextq_u8(lhs, rhs, I);
                }
                else
                {
                    return extract_pair(lhs, rhs, n, ::xsimd::detail::index_sequence<Is...>());
                }
            }

            template <class A, class T, size_t I, size_t... Is, detail::enable_sized_signed_t<T, 1> = 0>
            batch<T, A> extract_pair(batch<T, A> const& lhs, batch<T, A> const& rhs, std::size_t n, ::xsimd::detail::index_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vextq_s8(lhs, rhs, I);
                }
                else
                {
                    return extract_pair(lhs, rhs, n, ::xsimd::detail::index_sequence<Is...>());
                }
            }

            template <class A, class T, size_t I, size_t... Is, detail::enable_sized_unsigned_t<T, 2> = 0>
            batch<T, A> extract_pair(batch<T, A> const& lhs, batch<T, A> const& rhs, std::size_t n, ::xsimd::detail::index_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vextq_u16(lhs, rhs, I);
                }
                else
                {
                    return extract_pair(lhs, rhs, n, ::xsimd::detail::index_sequence<Is...>());
                }
            }

            template <class A, class T, size_t I, size_t... Is, detail::enable_sized_signed_t<T, 2> = 0>
            batch<T, A> extract_pair(batch<T, A> const& lhs, batch<T, A> const& rhs, std::size_t n, ::xsimd::detail::index_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vextq_s16(lhs, rhs, I);
                }
                else
                {
                    return extract_pair(lhs, rhs, n, ::xsimd::detail::index_sequence<Is...>());
                }
            }

            template <class A, class T, size_t I, size_t... Is, detail::enable_sized_unsigned_t<T, 4> = 0>
            batch<T, A> extract_pair(batch<T, A> const& lhs, batch<T, A> const& rhs, std::size_t n, ::xsimd::detail::index_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vextq_u32(lhs, rhs, I);
                }
                else
                {
                    return extract_pair(lhs, rhs, n, ::xsimd::detail::index_sequence<Is...>());
                }
            }

            template <class A, class T, size_t I, size_t... Is, detail::enable_sized_signed_t<T, 4> = 0>
            batch<T, A> extract_pair(batch<T, A> const& lhs, batch<T, A> const& rhs, std::size_t n, ::xsimd::detail::index_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vextq_s32(lhs, rhs, I);
                }
                else
                {
                    return extract_pair(lhs, rhs, n, ::xsimd::detail::index_sequence<Is...>());
                }
            }

            template <class A, class T, size_t I, size_t... Is, detail::enable_sized_unsigned_t<T, 8> = 0>
            batch<T, A> extract_pair(batch<T, A> const& lhs, batch<T, A> const& rhs, std::size_t n, ::xsimd::detail::index_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vextq_u64(lhs, rhs, I);
                }
                else
                {
                    return extract_pair(lhs, rhs, n, ::xsimd::detail::index_sequence<Is...>());
                }
            }

            template <class A, class T, size_t I, size_t... Is, detail::enable_sized_signed_t<T, 8> = 0>
            batch<T, A> extract_pair(batch<T, A> const& lhs, batch<T, A> const& rhs, std::size_t n, ::xsimd::detail::index_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vextq_s64(lhs, rhs, I);
                }
                else
                {
                    return extract_pair(lhs, rhs, n, ::xsimd::detail::index_sequence<Is...>());
                }
            }

            template <class A, size_t I, size_t... Is>
            batch<float, A> extract_pair(batch<float, A> const& lhs, batch<float, A> const& rhs, std::size_t n, ::xsimd::detail::index_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vextq_f32(lhs, rhs, I);
                }
                else
                {
                    return extract_pair(lhs, rhs, n, ::xsimd::detail::index_sequence<Is...>());
                }
            }

            template <class A, class T, size_t... Is>
            batch<T, A> extract_pair_impl(batch<T, A> const& lhs, batch<T, A> const& rhs, std::size_t n, ::xsimd::detail::index_sequence<0, Is...>)
            {
                if (n == 0)
                {
                    return rhs;
                }
                else
                {
                    return extract_pair(lhs, rhs, n, ::xsimd::detail::index_sequence<Is...>());
                }
            }
        }

        template <class A, class T>
        batch<T, A> extract_pair(batch<T, A> const& lhs, batch<T, A> const& rhs, std::size_t n, requires_arch<neon>)
        {
            constexpr std::size_t size = batch<T, A>::size;
            assert(0<= n && n< size && "index in bounds");
            return detail::extract_pair_impl(lhs, rhs, n, ::xsimd::detail::make_index_sequence<size>());
        }

        /******************
         * bitwise_lshift *
         ******************/

        namespace detail
        {
            template <class A, class T>
            batch<T, A> bitwise_lshift(batch<T, A> const& /*lhs*/, int /*n*/, ::xsimd::detail::int_sequence<>)
            {
                assert(false && "bitwise_lshift out of bounds");
                return batch<T, A>{};
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_unsigned_t<T, 1> = 0>
            batch<T, A> bitwise_lshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshlq_n_u8(lhs, I);
                }
                else
                {
                    return bitwise_lshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_signed_t<T, 1> = 0>
            batch<T, A> bitwise_lshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshlq_n_s8(lhs, I);
                }
                else
                {
                    return bitwise_lshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_unsigned_t<T, 2> = 0>
            batch<T, A> bitwise_lshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshlq_n_u16(lhs, I);
                }
                else
                {
                    return bitwise_lshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_signed_t<T, 2> = 0>
            batch<T, A> bitwise_lshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshlq_n_s16(lhs, I);
                }
                else
                {
                    return bitwise_lshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_unsigned_t<T, 4> = 0>
            batch<T, A> bitwise_lshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshlq_n_u32(lhs, I);
                }
                else
                {
                    return bitwise_lshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_signed_t<T, 4> = 0>
            batch<T, A> bitwise_lshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshlq_n_s32(lhs, I);
                }
                else
                {
                    return bitwise_lshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_unsigned_t<T, 8> = 0>
            batch<T, A> bitwise_lshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshlq_n_u64(lhs, I);
                }
                else
                {
                    return bitwise_lshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_signed_t<T, 8> = 0>
            batch<T, A> bitwise_lshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshlq_n_s64(lhs, I);
                }
                else
                {
                    return bitwise_lshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int... Is>
            batch<T, A> bitwise_lshift_impl(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<0, Is...>)
            {
                if (n == 0)
                {
                    return lhs;
                }
                else
                {
                    return bitwise_lshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }
        }
        
        template <class A, class T>
        batch<T, A> bitwise_lshift(batch<T, A> const& lhs, int n, requires_arch<neon>)
        {
            constexpr std::size_t size = sizeof(typename batch<T, A>::value_type) * 8;
            assert(0<= n && n< size && "index in bounds");
            return detail::bitwise_lshift_impl(lhs, n, ::xsimd::detail::make_int_sequence<size>());
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 1> = 0>
        batch<T, A> bitwise_lshift(batch<T, A> const& lhs, batch<as_signed_integer_t<T>, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_u8(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 1> = 0>
        batch<T, A> bitwise_lshift(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_s8(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 2> = 0>
        batch<T, A> bitwise_lshift(batch<T, A> const& lhs, batch<as_signed_integer_t<T>, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_u16(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 2> = 0>
        batch<T, A> bitwise_lshift(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_s16(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 4> = 0>
        batch<T, A> bitwise_lshift(batch<T, A> const& lhs, batch<as_signed_integer_t<T>, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_u32(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 4> = 0>
        batch<T, A> bitwise_lshift(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_s32(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 8> = 0>
        batch<T, A> bitwise_lshift(batch<T, A> const& lhs, batch<as_signed_integer_t<T>, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_u64(lhs, rhs);
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 8> = 0>
        batch<T, A> bitwise_lshift(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_s64(lhs, rhs);
        }

        /******************
         * bitwise_rshift *
         ******************/

        namespace detail
        {
            template <class A, class T>
            batch<T, A> bitwise_rshift(batch<T, A> const& /*lhs*/, int /*n*/, ::xsimd::detail::int_sequence<>)
            {
                assert(false && "bitwise_rshift out of bounds");
                return batch<T, A>{};
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_unsigned_t<T, 1> = 0>
            batch<T, A> bitwise_rshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshrq_n_u8(lhs, I);
                }
                else
                {
                    return bitwise_rshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_signed_t<T, 1> = 0>
            batch<T, A> bitwise_rshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshrq_n_s8(lhs, I);
                }
                else
                {
                    return bitwise_rshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_unsigned_t<T, 2> = 0>
            batch<T, A> bitwise_rshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshrq_n_u16(lhs, I);
                }
                else
                {
                    return bitwise_rshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_signed_t<T, 2> = 0>
            batch<T, A> bitwise_rshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshrq_n_s16(lhs, I);
                }
                else
                {
                    return bitwise_rshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_unsigned_t<T, 4> = 0>
            batch<T, A> bitwise_rshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshrq_n_u32(lhs, I);
                }
                else
                {
                    return bitwise_rshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_signed_t<T, 4> = 0>
            batch<T, A> bitwise_rshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshrq_n_s32(lhs, I);
                }
                else
                {
                    return bitwise_rshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_unsigned_t<T, 8> = 0>
            batch<T, A> bitwise_rshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshrq_n_u64(lhs, I);
                }
                else
                {
                    return bitwise_rshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int I, int... Is, detail::enable_sized_signed_t<T, 8> = 0>
            batch<T, A> bitwise_rshift(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<I, Is...>)
            {
                if (n == I)
                {
                    return vshrq_n_s64(lhs, I);
                }
                else
                {
                    return bitwise_rshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }

            template <class A, class T, int... Is>
            batch<T, A> bitwise_rshift_impl(batch<T, A> const& lhs, int n, ::xsimd::detail::int_sequence<0, Is...>)
            {
                if (n == 0)
                {
                    return lhs;
                }
                else
                {
                    return bitwise_rshift(lhs, n, ::xsimd::detail::int_sequence<Is...>());
                }
            }
        }
        
        template <class A, class T>
        batch<T, A> bitwise_rshift(batch<T, A> const& lhs, int n, requires_arch<neon>)
        {
            constexpr std::size_t size = sizeof(typename batch<T, A>::value_type) * 8;
            assert(0<= n && n< size && "index in bounds");
            return detail::bitwise_rshift_impl(lhs, n, ::xsimd::detail::make_int_sequence<size>());
        }
        
        template <class A, class T, detail::enable_sized_unsigned_t<T, 1> = 0>
        batch<T, A> bitwise_rshift(batch<T, A> const& lhs, batch<as_signed_integer_t<T>, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_u8(lhs, vnegq_s8(rhs));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 1> = 0>
        batch<T, A> bitwise_rshift(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_s8(lhs, vnegq_s8(rhs));
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 2> = 0>
        batch<T, A> bitwise_rshift(batch<T, A> const& lhs, batch<as_signed_integer_t<T>, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_u16(lhs, vnegq_s16(rhs));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 2> = 0>
        batch<T, A> bitwise_rshift(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_s16(lhs, vnegq_s16(rhs));
        }

        template <class A, class T, detail::enable_sized_unsigned_t<T, 4> = 0>
        batch<T, A> bitwise_rshift(batch<T, A> const& lhs, batch<as_signed_integer_t<T>, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_u32(lhs, vnegq_s32(rhs));
        }

        template <class A, class T, detail::enable_sized_signed_t<T, 4> = 0>
        batch<T, A> bitwise_rshift(batch<T, A> const& lhs, batch<T, A> const& rhs, requires_arch<neon>)
        {
            return vshlq_s32(lhs, vnegq_s32(rhs));
        }

        // Overloads of bitwise shifts accepting two batches of uint64/int64 are not available with ARMv7

        /*******
         * all *
         *******/

        template <class A, class T, detail::enable_sized_t<T, 1> = 0>
        bool all(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            uint8x8_t tmp = vand_u8(vget_low_u8(arg), vget_high_u8(arg));
            tmp = vpmin_u8(tmp, tmp);
            tmp = vpmin_u8(tmp, tmp);
            tmp = vpmin_u8(tmp, tmp);
            return vget_lane_u8(tmp, 0);
        }

        template <class A, class T, detail::enable_sized_t<T, 2> = 0>
        bool all(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            uint16x4_t tmp = vand_u16(vget_low_u16(arg), vget_high_u16(arg));
            tmp = vpmin_u16(tmp, tmp);
            tmp = vpmin_u16(tmp, tmp);
            return vget_lane_u16(tmp, 0) != 0;
        }

        template <class A, class T, detail::enable_sized_t<T, 4> = 0>
        bool all(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            uint32x2_t tmp = vand_u32(vget_low_u32(arg), vget_high_u32(arg));
            return vget_lane_u32(vpmin_u32(tmp, tmp), 0) != 0;
        }

        template <class A, class T, detail::enable_sized_t<T, 8> = 0>
        bool all(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            uint64x1_t tmp = vand_u64(vget_low_u64(arg), vget_high_u64(arg));
            return vget_lane_u64(tmp, 0) != 0;
        }

        /*******
         * any *
         *******/

        template <class A, class T, detail::enable_sized_t<T, 1> = 0>
        bool any(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            uint8x8_t tmp = vorr_u8(vget_low_u8(arg), vget_high_u8(arg));
            tmp = vpmax_u8(tmp, tmp);
            tmp = vpmax_u8(tmp, tmp);
            tmp = vpmax_u8(tmp, tmp);
            return vget_lane_u8(tmp, 0);
        }

        template <class A, class T, detail::enable_sized_t<T, 2> = 0>
        bool any(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            uint16x4_t tmp = vorr_u16(vget_low_u16(arg), vget_high_u16(arg));
            tmp = vpmax_u16(tmp, tmp);
            tmp = vpmax_u16(tmp, tmp);
            return vget_lane_u16(tmp, 0);
        }

        template <class A, class T, detail::enable_sized_t<T, 4> = 0>
        bool any(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            uint32x2_t tmp = vorr_u32(vget_low_u32(arg), vget_high_u32(arg));
            return vget_lane_u32(vpmax_u32(tmp, tmp), 0);
        }

        template <class A, class T, detail::enable_sized_t<T, 8> = 0>
        bool any(batch_bool<T, A> const& arg, requires_arch<neon>)
        {
            uint64x1_t tmp = vorr_u64(vget_low_u64(arg), vget_high_u64(arg));
            return bool(vget_lane_u64(tmp, 0));
        }

        /****************
         * bitwise_cast *
         ****************/

        namespace detail
        {
            template <class R, class... T>
            struct bitwise_caster_impl
            {
                using container_type = std::tuple<R (*)(T)...>;
                container_type m_func;

                template <class U>
                R apply(U rhs) const
                {
                    using func_type = R (*)(U);
                    auto func = xsimd::detail::get<func_type>(m_func);
                    return func(rhs);
                }
            };

            template <class R, class... T>
            constexpr bitwise_caster_impl<R, T...> make_bitwise_caster_impl(R (*...arg)(T))
            {
                return {std::make_tuple(arg...)};
            }

            template <class... T>
            struct type_list {};
            
            template <class RTL, class TTL>
            struct bitwise_caster;

            template <class... R, class... T>
            struct bitwise_caster<type_list<R...>, type_list<T...>>
            {
                using container_type = std::tuple<bitwise_caster_impl<R, T...>...>;
                container_type m_caster;

                template <class V, class U>
                V apply(U rhs) const
                {
                    using caster_type = bitwise_caster_impl<V, T...>;
                    auto caster = xsimd::detail::get<caster_type>(m_caster);
                    return caster.apply(rhs);
                }
            };

            template <class... T>
            using bitwise_caster_t = bitwise_caster<type_list<T...>, type_list<T...>>;
                    
            using neon_bitwise_caster = bitwise_caster_t<uint8x16_t, int8x16_t,
                                                        uint16x8_t, int16x8_t,
                                                        uint32x4_t, int32x4_t,
                                                        uint64x2_t, int64x2_t,
                                                        float32x4_t>;

            inline uint8x16_t identity_u8(uint8x16_t arg) { return arg; }
            inline int8x16_t identity_s8(int8x16_t arg) { return arg; }
            inline uint16x8_t identity_u16(uint16x8_t arg) { return arg; }
            inline int16x8_t identity_s16(int16x8_t arg) { return arg; }
            inline uint32x4_t identity_u32(uint32x4_t arg) { return arg; }
            inline int32x4_t identity_s32(int32x4_t arg) { return arg; }
            inline uint64x2_t identity_u64(uint64x2_t arg) { return arg; }
            inline int64x2_t identity_s64(int64x2_t arg) { return arg; }
            inline float32x4_t identity_f32(float32x4_t arg) { return arg; }
        }

        template <class A, class T, class R>
        batch<R, A> bitwise_cast(batch<T, A> const& arg, batch<R, A> const&, requires_arch<neon>)
        {
            constexpr detail::neon_bitwise_caster caster = {
                std::make_tuple(
                detail::make_bitwise_caster_impl(detail::identity_u8,  vreinterpretq_u8_s8,  vreinterpretq_u8_u16, vreinterpretq_u8_s16,
                                                 vreinterpretq_u8_u32, vreinterpretq_u8_s32, vreinterpretq_u8_u64, vreinterpretq_u8_s64,
                                                 vreinterpretq_u8_f32),
                detail::make_bitwise_caster_impl(vreinterpretq_s8_u8,   detail::identity_s8,  vreinterpretq_s8_u16, vreinterpretq_s8_s16,
                                                 vreinterpretq_s8_u32, vreinterpretq_s8_s32, vreinterpretq_s8_u64, vreinterpretq_s8_s64,
                                                 vreinterpretq_s8_f32),
                detail::make_bitwise_caster_impl(vreinterpretq_u16_u8,  vreinterpretq_u16_s8,  detail::identity_u16,  vreinterpretq_u16_s16,
                                                 vreinterpretq_u16_u32, vreinterpretq_u16_s32, vreinterpretq_u16_u64, vreinterpretq_u16_s64,
                                                 vreinterpretq_u16_f32),
                detail::make_bitwise_caster_impl(vreinterpretq_s16_u8,  vreinterpretq_s16_s8,  vreinterpretq_s16_u16, detail::identity_s16,
                                                 vreinterpretq_s16_u32, vreinterpretq_s16_s32, vreinterpretq_s16_u64, vreinterpretq_s16_s64,
                                                 vreinterpretq_s16_f32),
                detail::make_bitwise_caster_impl(vreinterpretq_u32_u8,  vreinterpretq_u32_s8,  vreinterpretq_u32_u16, vreinterpretq_u32_s16,
                                                 detail::identity_u32, vreinterpretq_u32_s32, vreinterpretq_u32_u64, vreinterpretq_u32_s64,
                                                 vreinterpretq_u32_f32),
                detail::make_bitwise_caster_impl(vreinterpretq_s32_u8,  vreinterpretq_s32_s8,  vreinterpretq_s32_u16, vreinterpretq_s32_s16,
                                                 vreinterpretq_s32_u32, detail::identity_s32,  vreinterpretq_s32_u64, vreinterpretq_s32_s64,
                                                 vreinterpretq_s32_f32),
                detail::make_bitwise_caster_impl(vreinterpretq_u64_u8,  vreinterpretq_u64_s8,  vreinterpretq_u64_u16, vreinterpretq_u64_s16,
                                                 vreinterpretq_u64_u32, vreinterpretq_u64_s32, detail::identity_u64,  vreinterpretq_u64_s64,
                                                 vreinterpretq_u64_f32),
                detail::make_bitwise_caster_impl(vreinterpretq_s64_u8,  vreinterpretq_s64_s8,  vreinterpretq_s64_u16, vreinterpretq_s64_s16,
                                                 vreinterpretq_s64_u32, vreinterpretq_s64_s32, vreinterpretq_s64_u64, detail::identity_s64,
                                                 vreinterpretq_s64_f32),
                detail::make_bitwise_caster_impl(vreinterpretq_f32_u8,  vreinterpretq_f32_s8,  vreinterpretq_f32_u16, vreinterpretq_f32_s16,
                                                 vreinterpretq_f32_u32, vreinterpretq_f32_s32, vreinterpretq_f32_u64, vreinterpretq_f32_s64,
                                                 detail::identity_f32))
            };
            using src_register_type = typename batch<T, A>::register_type;
            using dst_register_type = typename batch<R, A>::register_type;
            return caster.apply<dst_register_type>(src_register_type(arg));
        }

        /*************
         * bool_cast *
         *************/

        template <class A>
        batch_bool<float, A> bool_cast(batch_bool<int32_t, A> const& arg, requires_arch<neon>)
        {
            using register_type = typename batch_bool<int32_t, A>::register_type;
            return register_type(arg);
        }

        template <class A>
        batch_bool<int32_t, A> bool_cast(batch_bool<float, A> const& arg, requires_arch<neon>)
        {
            using register_type = typename batch_bool<float, A>::register_type;
            return register_type(arg);
        }

        /**********
         * to_int *
         **********/

        template <class A>
        batch<int32_t, A> to_int(const batch<float, A>& x, requires_arch<neon>)
        {
            return vcvtq_s32_f32(x);
        }

        /************
         * to_float *
         ************/

        template <class A>
        batch<float, A> to_float(const batch<int32_t, A>& x, requires_arch<neon>)
        {
            return vcvtq_f32_s32(x);
        }

        /*************
         * fast_cast *
         *************/

        namespace detail
        {
            template <class Tin, class Tout, class A>
            batch<Tout, A> fast_cast(batch<Tin, A> const& in, batch<Tout, A> const& out, requires_arch<neon>)
            {
                return bitwise_cast(in, out, A{});
            }
        }

        /*********
         * isnan *
         *********/

        template <class A>
        batch_bool<float, A> isnan(batch<float, A> const& arg, requires_arch<neon>)
        {
            return !(arg == arg);
        }
    }
}

#endif
