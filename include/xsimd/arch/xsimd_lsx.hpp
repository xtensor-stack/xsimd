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

#ifndef XSIMD_LSX_HPP
#define XSIMD_LSX_HPP

#include "../types/xsimd_batch_constant.hpp"
#include "../types/xsimd_lsx_register.hpp"
#include "../types/xsimd_utils.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace xsimd
{
    namespace kernel
    {
        namespace detail
        {
            template <class T>
            struct lsx_set_vector
            {
                typedef T type __attribute__((vector_size(16)));
            };

            template <class T, class A>
            XSIMD_INLINE __m128i lsx_to_int(batch<T, A> const& value) noexcept
            {
                return bit_cast<__m128i>(value.data);
            }

            template <class T, class A>
            XSIMD_INLINE typename batch<T, A>::register_type lsx_from_int(__m128i value) noexcept
            {
                return bit_cast<typename batch<T, A>::register_type>(value);
            }

            template <class T>
            XSIMD_INLINE std::uint32_t lsx_mask(__m128i value) noexcept
            {
                if constexpr (sizeof(T) == 1)
                {
                    return static_cast<std::uint32_t>(__lsx_vpickve2gr_w(__lsx_vmskltz_b(value), 0));
                }
                else if constexpr (sizeof(T) == 2)
                {
                    return static_cast<std::uint32_t>(__lsx_vpickve2gr_w(__lsx_vmskltz_h(value), 0));
                }
                else if constexpr (sizeof(T) == 4)
                {
                    return static_cast<std::uint32_t>(__lsx_vpickve2gr_w(__lsx_vmskltz_w(value), 0));
                }
                else
                {
                    return static_cast<std::uint32_t>(__lsx_vpickve2gr_w(__lsx_vmskltz_d(value), 0));
                }
            }

            // fast_cast
            template <class A>
            XSIMD_INLINE batch<float, A> fast_cast(batch<std::int32_t, A> const& self, batch<float, A> const&, requires_arch<lsx>) noexcept
            {
                return __lsx_vffint_s_w(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<float, A> fast_cast(batch<std::uint32_t, A> const& self, batch<float, A> const&, requires_arch<lsx>) noexcept
            {
                return __lsx_vffint_s_wu(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<double, A> fast_cast(batch<std::int64_t, A> const& self, batch<double, A> const&, requires_arch<lsx>) noexcept
            {
                return __lsx_vffint_d_l(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<double, A> fast_cast(batch<std::uint64_t, A> const& self, batch<double, A> const&, requires_arch<lsx>) noexcept
            {
                return __lsx_vffint_d_lu(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<std::int32_t, A> fast_cast(batch<float, A> const& self, batch<std::int32_t, A> const&, requires_arch<lsx>) noexcept
            {
                return __lsx_vftintrz_w_s(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<std::uint32_t, A> fast_cast(batch<float, A> const& self, batch<std::uint32_t, A> const&, requires_arch<lsx>) noexcept
            {
                return __lsx_vftintrz_wu_s(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<std::int64_t, A> fast_cast(batch<double, A> const& self, batch<std::int64_t, A> const&, requires_arch<lsx>) noexcept
            {
                return __lsx_vftintrz_l_d(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<std::uint64_t, A> fast_cast(batch<double, A> const& self, batch<std::uint64_t, A> const&, requires_arch<lsx>) noexcept
            {
                return __lsx_vftintrz_lu_d(self.data);
            }
        }

        // abs
        template <class A>
        XSIMD_INLINE batch<float, A> abs(batch<float, A> const& self, requires_arch<lsx>) noexcept
        {
            return detail::lsx_from_int<float, A>(__lsx_vbitclri_w(detail::lsx_to_int(self), 31));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> abs(batch<double, A> const& self, requires_arch<lsx>) noexcept
        {
            return detail::lsx_from_int<double, A>(__lsx_vbitclri_d(detail::lsx_to_int(self), 63));
        }

        // add
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> add(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lsx_vadd_b(self.data, other.data);
            else if constexpr (sizeof(T) == 2)
                return __lsx_vadd_h(self.data, other.data);
            else if constexpr (sizeof(T) == 4)
                return __lsx_vadd_w(self.data, other.data);
            else
                return __lsx_vadd_d(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<float, A> add(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfadd_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<double, A> add(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfadd_d(self.data, other.data);
        }

        // all/any
        template <class A, class T>
        XSIMD_INLINE bool all(batch_bool<T, A> const& self, requires_arch<lsx>) noexcept
        {
            constexpr std::uint32_t all_bits = (std::uint32_t(1) << batch_bool<T, A>::size) - 1;
            return detail::lsx_mask<T>(self.data) == all_bits;
        }

        template <class A, class T>
        XSIMD_INLINE bool any(batch_bool<T, A> const& self, requires_arch<lsx>) noexcept
        {
            return detail::lsx_mask<T>(self.data) != 0;
        }

        // batch_bool_cast
        template <class A, class T_out, class T_in>
        XSIMD_INLINE batch_bool<T_out, A> batch_bool_cast(batch_bool<T_in, A> const& self, batch_bool<T_out, A> const&, requires_arch<lsx>) noexcept
        {
            return self.data;
        }

        // bitwise operations
        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_and(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            return detail::lsx_from_int<T, A>(__lsx_vand_v(detail::lsx_to_int(self), detail::lsx_to_int(other)));
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_and(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vand_v(self.data, other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_andnot(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            return detail::lsx_from_int<T, A>(__lsx_vandn_v(detail::lsx_to_int(other), detail::lsx_to_int(self)));
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_andnot(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vandn_v(other.data, self.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_not(batch<T, A> const& self, requires_arch<lsx>) noexcept
        {
            const __m128i bits = detail::lsx_to_int(self);
            return detail::lsx_from_int<T, A>(__lsx_vnor_v(bits, bits));
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_not(batch_bool<T, A> const& self, requires_arch<lsx>) noexcept
        {
            return __lsx_vnor_v(self.data, self.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_or(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            return detail::lsx_from_int<T, A>(__lsx_vor_v(detail::lsx_to_int(self), detail::lsx_to_int(other)));
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_or(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vor_v(self.data, other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_xor(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            return detail::lsx_from_int<T, A>(__lsx_vxor_v(detail::lsx_to_int(self), detail::lsx_to_int(other)));
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_xor(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vxor_v(self.data, other.data);
        }

        template <class A, class T_in, class T_out>
        XSIMD_INLINE batch<T_out, A> bitwise_cast(batch<T_in, A> const& self, batch<T_out, A> const&, requires_arch<lsx>) noexcept
        {
            return bit_cast<typename batch<T_out, A>::register_type>(self.data);
        }

        // shifts
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> bitwise_lshift(batch<T, A> const& self, std::int32_t other, requires_arch<lsx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lsx_vsll_b(self.data, __lsx_vreplgr2vr_b(other));
            else if constexpr (sizeof(T) == 2)
                return __lsx_vsll_h(self.data, __lsx_vreplgr2vr_h(other));
            else if constexpr (sizeof(T) == 4)
                return __lsx_vsll_w(self.data, __lsx_vreplgr2vr_w(other));
            else
                return __lsx_vsll_d(self.data, __lsx_vreplgr2vr_d(other));
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> bitwise_rshift(batch<T, A> const& self, std::int32_t other, requires_arch<lsx>) noexcept
        {
            if constexpr (std::is_signed_v<T>)
            {
                if constexpr (sizeof(T) == 1)
                    return __lsx_vsra_b(self.data, __lsx_vreplgr2vr_b(other));
                else if constexpr (sizeof(T) == 2)
                    return __lsx_vsra_h(self.data, __lsx_vreplgr2vr_h(other));
                else if constexpr (sizeof(T) == 4)
                    return __lsx_vsra_w(self.data, __lsx_vreplgr2vr_w(other));
                else
                    return __lsx_vsra_d(self.data, __lsx_vreplgr2vr_d(other));
            }
            else
            {
                if constexpr (sizeof(T) == 1)
                    return __lsx_vsrl_b(self.data, __lsx_vreplgr2vr_b(other));
                else if constexpr (sizeof(T) == 2)
                    return __lsx_vsrl_h(self.data, __lsx_vreplgr2vr_h(other));
                else if constexpr (sizeof(T) == 4)
                    return __lsx_vsrl_w(self.data, __lsx_vreplgr2vr_w(other));
                else
                    return __lsx_vsrl_d(self.data, __lsx_vreplgr2vr_d(other));
            }
        }

        // div
        template <class A>
        XSIMD_INLINE batch<float, A> div(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfdiv_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<double, A> div(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfdiv_d(self.data, other.data);
        }

        // broadcast
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> broadcast(T value, requires_arch<lsx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lsx_vreplgr2vr_b(static_cast<int>(value));
            else if constexpr (sizeof(T) == 2)
                return __lsx_vreplgr2vr_h(static_cast<int>(value));
            else if constexpr (sizeof(T) == 4)
                return __lsx_vreplgr2vr_w(bit_cast<sized_int_t<sizeof(T)>>(value));
            else
                return __lsx_vreplgr2vr_d(bit_cast<sized_int_t<sizeof(T)>>(value));
        }

        template <class A>
        XSIMD_INLINE batch<float, A> broadcast(float value, requires_arch<lsx>) noexcept
        {
            const auto bits = bit_cast<std::int32_t>(value);
            return bit_cast<__m128>(__lsx_vreplgr2vr_w(bits));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> broadcast(double value, requires_arch<lsx>) noexcept
        {
            const auto bits = bit_cast<std::int64_t>(value);
            return bit_cast<__m128d>(__lsx_vreplgr2vr_d(static_cast<long>(bits)));
        }

        // comparisons
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lsx_vseq_b(self.data, other.data);
            else if constexpr (sizeof(T) == 2)
                return __lsx_vseq_h(self.data, other.data);
            else if constexpr (sizeof(T) == 4)
                return __lsx_vseq_w(self.data, other.data);
            else
                return __lsx_vseq_d(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch_bool<float, A> eq(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfcmp_ceq_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch_bool<double, A> eq(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfcmp_ceq_d(self.data, other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> eq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<lsx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lsx_vseq_b(self.data, other.data);
            else if constexpr (sizeof(T) == 2)
                return __lsx_vseq_h(self.data, other.data);
            else if constexpr (sizeof(T) == 4)
                return __lsx_vseq_w(self.data, other.data);
            else
                return __lsx_vseq_d(self.data, other.data);
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch_bool<T, A> lt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            if constexpr (std::is_signed_v<T>)
            {
                if constexpr (sizeof(T) == 1)
                    return __lsx_vslt_b(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lsx_vslt_h(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lsx_vslt_w(self.data, other.data);
                else
                    return __lsx_vslt_d(self.data, other.data);
            }
            else
            {
                if constexpr (sizeof(T) == 1)
                    return __lsx_vslt_bu(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lsx_vslt_hu(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lsx_vslt_wu(self.data, other.data);
                else
                    return __lsx_vslt_du(self.data, other.data);
            }
        }

        template <class A>
        XSIMD_INLINE batch_bool<float, A> lt(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfcmp_clt_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch_bool<double, A> lt(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfcmp_clt_d(self.data, other.data);
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch_bool<T, A> le(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            if constexpr (std::is_signed_v<T>)
            {
                if constexpr (sizeof(T) == 1)
                    return __lsx_vsle_b(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lsx_vsle_h(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lsx_vsle_w(self.data, other.data);
                else
                    return __lsx_vsle_d(self.data, other.data);
            }
            else
            {
                if constexpr (sizeof(T) == 1)
                    return __lsx_vsle_bu(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lsx_vsle_hu(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lsx_vsle_wu(self.data, other.data);
                else
                    return __lsx_vsle_du(self.data, other.data);
            }
        }

        template <class A>
        XSIMD_INLINE batch_bool<float, A> le(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfcmp_cle_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch_bool<double, A> le(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfcmp_cle_d(self.data, other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            return lt(other, self, lsx {});
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> ge(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            return le(other, self, lsx {});
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            const __m128i equal = eq(self, other, lsx {}).data;
            return __lsx_vnor_v(equal, equal);
        }

        template <class A>
        XSIMD_INLINE batch_bool<float, A> neq(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfcmp_cune_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch_bool<double, A> neq(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfcmp_cune_d(self.data, other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> neq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vxor_v(self.data, other.data);
        }

        // first
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE T first(batch<T, A> const& self, requires_arch<lsx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
            {
                if constexpr (std::is_signed_v<T>)
                    return static_cast<T>(__lsx_vpickve2gr_b(self.data, 0));
                else
                    return static_cast<T>(__lsx_vpickve2gr_bu(self.data, 0));
            }
            else if constexpr (sizeof(T) == 2)
            {
                if constexpr (std::is_signed_v<T>)
                    return static_cast<T>(__lsx_vpickve2gr_h(self.data, 0));
                else
                    return static_cast<T>(__lsx_vpickve2gr_hu(self.data, 0));
            }
            else if constexpr (sizeof(T) == 4)
            {
                if constexpr (std::is_signed_v<T>)
                    return static_cast<T>(__lsx_vpickve2gr_w(self.data, 0));
                else
                    return static_cast<T>(__lsx_vpickve2gr_wu(self.data, 0));
            }
            else
            {
                if constexpr (std::is_signed_v<T>)
                    return static_cast<T>(__lsx_vpickve2gr_d(self.data, 0));
                else
                    return static_cast<T>(__lsx_vpickve2gr_du(self.data, 0));
            }
        }

        template <class A>
        XSIMD_INLINE float first(batch<float, A> const& self, requires_arch<lsx>) noexcept
        {
            const auto bits = static_cast<std::uint32_t>(__lsx_vpickve2gr_w(detail::lsx_to_int(self), 0));
            return bit_cast<float>(bits);
        }

        template <class A>
        XSIMD_INLINE double first(batch<double, A> const& self, requires_arch<lsx>) noexcept
        {
            const auto bits = static_cast<std::uint64_t>(__lsx_vpickve2gr_du(detail::lsx_to_int(self), 0));
            return bit_cast<double>(bits);
        }

        template <class A, class T>
        XSIMD_INLINE std::complex<T> first(batch<std::complex<T>, A> const& self, requires_arch<lsx>) noexcept
        {
            return { first(self.real(), lsx {}), first(self.imag(), lsx {}) };
        }

        // horizontal add of rows
        template <class A>
        XSIMD_INLINE batch<float, A> haddp(batch<float, A> const* row, requires_arch<lsx>) noexcept
        {
            const __m128i ab_lo = __lsx_vilvl_w(detail::lsx_to_int(row[1]), detail::lsx_to_int(row[0]));
            const __m128i ab_hi = __lsx_vilvh_w(detail::lsx_to_int(row[1]), detail::lsx_to_int(row[0]));
            const __m128 ab = __lsx_vfadd_s(bit_cast<__m128>(ab_lo), bit_cast<__m128>(ab_hi));
            const __m128i cd_lo = __lsx_vilvl_w(detail::lsx_to_int(row[3]), detail::lsx_to_int(row[2]));
            const __m128i cd_hi = __lsx_vilvh_w(detail::lsx_to_int(row[3]), detail::lsx_to_int(row[2]));
            const __m128 cd = __lsx_vfadd_s(bit_cast<__m128>(cd_lo), bit_cast<__m128>(cd_hi));
            const __m128i lo = __lsx_vilvl_d(bit_cast<__m128i>(cd), bit_cast<__m128i>(ab));
            const __m128i hi = __lsx_vilvh_d(bit_cast<__m128i>(cd), bit_cast<__m128i>(ab));
            return __lsx_vfadd_s(bit_cast<__m128>(lo), bit_cast<__m128>(hi));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> haddp(batch<double, A> const* row, requires_arch<lsx>) noexcept
        {
            const __m128i lo = __lsx_vilvl_d(detail::lsx_to_int(row[1]), detail::lsx_to_int(row[0]));
            const __m128i hi = __lsx_vilvh_d(detail::lsx_to_int(row[1]), detail::lsx_to_int(row[0]));
            return __lsx_vfadd_d(bit_cast<__m128d>(lo), bit_cast<__m128d>(hi));
        }

        // load
        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE batch<T, A> load_unaligned(T const* mem, convert<T>, requires_arch<lsx>) noexcept
        {
            return detail::lsx_from_int<T, A>(__lsx_vld(mem, 0));
        }

        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE batch<T, A> load_aligned(T const* mem, convert<T>, requires_arch<lsx>) noexcept
        {
            return detail::lsx_from_int<T, A>(__lsx_vld(mem, 0));
        }

        // load/store complex helpers
        namespace detail
        {
            template <class A, class T>
            XSIMD_INLINE batch<std::complex<T>, A> load_complex(batch<T, A> const& first_chunk, batch<T, A> const& second_chunk, requires_arch<lsx>) noexcept
            {
                __m128i real;
                __m128i imag;
                if constexpr (sizeof(T) == 4)
                {
                    real = __lsx_vpickev_w(lsx_to_int(second_chunk), lsx_to_int(first_chunk));
                    imag = __lsx_vpickod_w(lsx_to_int(second_chunk), lsx_to_int(first_chunk));
                }
                else
                {
                    real = __lsx_vpickev_d(lsx_to_int(second_chunk), lsx_to_int(first_chunk));
                    imag = __lsx_vpickod_d(lsx_to_int(second_chunk), lsx_to_int(first_chunk));
                }
                return { lsx_from_int<T, A>(real), lsx_from_int<T, A>(imag) };
            }

            template <class A, class T>
            XSIMD_INLINE batch<T, A> complex_low(batch<std::complex<T>, A> const& self, requires_arch<lsx>) noexcept
            {
                if constexpr (sizeof(T) == 4)
                    return lsx_from_int<T, A>(__lsx_vilvl_w(lsx_to_int(self.imag()), lsx_to_int(self.real())));
                else
                    return lsx_from_int<T, A>(__lsx_vilvl_d(lsx_to_int(self.imag()), lsx_to_int(self.real())));
            }

            template <class A, class T>
            XSIMD_INLINE batch<T, A> complex_high(batch<std::complex<T>, A> const& self, requires_arch<lsx>) noexcept
            {
                if constexpr (sizeof(T) == 4)
                    return lsx_from_int<T, A>(__lsx_vilvh_w(lsx_to_int(self.imag()), lsx_to_int(self.real())));
                else
                    return lsx_from_int<T, A>(__lsx_vilvh_d(lsx_to_int(self.imag()), lsx_to_int(self.real())));
            }
        }

        // max/min
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            if constexpr (std::is_signed_v<T>)
            {
                if constexpr (sizeof(T) == 1)
                    return __lsx_vmax_b(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lsx_vmax_h(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lsx_vmax_w(self.data, other.data);
                else
                    return __lsx_vmax_d(self.data, other.data);
            }
            else
            {
                if constexpr (sizeof(T) == 1)
                    return __lsx_vmax_bu(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lsx_vmax_hu(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lsx_vmax_wu(self.data, other.data);
                else
                    return __lsx_vmax_du(self.data, other.data);
            }
        }

        template <class A>
        XSIMD_INLINE batch<float, A> max(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lsx>) noexcept
        {
            const __m128i cond = __lsx_vfcmp_clt_s(self.data, other.data);
            return detail::lsx_from_int<float, A>(__lsx_vbitsel_v(detail::lsx_to_int(self), detail::lsx_to_int(other), cond));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> max(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lsx>) noexcept
        {
            const __m128i cond = __lsx_vfcmp_clt_d(self.data, other.data);
            return detail::lsx_from_int<double, A>(__lsx_vbitsel_v(detail::lsx_to_int(self), detail::lsx_to_int(other), cond));
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            if constexpr (std::is_signed_v<T>)
            {
                if constexpr (sizeof(T) == 1)
                    return __lsx_vmin_b(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lsx_vmin_h(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lsx_vmin_w(self.data, other.data);
                else
                    return __lsx_vmin_d(self.data, other.data);
            }
            else
            {
                if constexpr (sizeof(T) == 1)
                    return __lsx_vmin_bu(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lsx_vmin_hu(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lsx_vmin_wu(self.data, other.data);
                else
                    return __lsx_vmin_du(self.data, other.data);
            }
        }

        template <class A>
        XSIMD_INLINE batch<float, A> min(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lsx>) noexcept
        {
            const __m128i cond = __lsx_vfcmp_clt_s(other.data, self.data);
            return detail::lsx_from_int<float, A>(__lsx_vbitsel_v(detail::lsx_to_int(self), detail::lsx_to_int(other), cond));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> min(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lsx>) noexcept
        {
            const __m128i cond = __lsx_vfcmp_clt_d(other.data, self.data);
            return detail::lsx_from_int<double, A>(__lsx_vbitsel_v(detail::lsx_to_int(self), detail::lsx_to_int(other), cond));
        }

        // mul/neg
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lsx_vmul_b(self.data, other.data);
            else if constexpr (sizeof(T) == 2)
                return __lsx_vmul_h(self.data, other.data);
            else if constexpr (sizeof(T) == 4)
                return __lsx_vmul_w(self.data, other.data);
            else
                return __lsx_vmul_d(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<float, A> mul(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfmul_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<double, A> mul(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfmul_d(self.data, other.data);
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> neg(batch<T, A> const& self, requires_arch<lsx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lsx_vneg_b(self.data);
            else if constexpr (sizeof(T) == 2)
                return __lsx_vneg_h(self.data);
            else if constexpr (sizeof(T) == 4)
                return __lsx_vneg_w(self.data);
            else
                return __lsx_vneg_d(self.data);
        }

        template <class A>
        XSIMD_INLINE batch<float, A> neg(batch<float, A> const& self, requires_arch<lsx>) noexcept
        {
            const auto sign_bit = bit_cast<std::int32_t>(std::uint32_t(1) << 31);
            const __m128i sign = __lsx_vreplgr2vr_w(sign_bit);
            return detail::lsx_from_int<float, A>(__lsx_vxor_v(detail::lsx_to_int(self), sign));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> neg(batch<double, A> const& self, requires_arch<lsx>) noexcept
        {
            const auto sign_bit = bit_cast<std::int64_t>(std::uint64_t(1) << 63);
            const __m128i sign = __lsx_vreplgr2vr_d(sign_bit);
            return detail::lsx_from_int<double, A>(__lsx_vxor_v(detail::lsx_to_int(self), sign));
        }

        // rsqrt/sqrt
        template <class A>
        XSIMD_INLINE batch<float, A> rsqrt(batch<float, A> const& self, requires_arch<lsx>) noexcept
        {
            return __lsx_vfrsqrt_s(self.data);
        }

        template <class A>
        XSIMD_INLINE batch<double, A> rsqrt(batch<double, A> const& self, requires_arch<lsx>) noexcept
        {
            return __lsx_vfrsqrt_d(self.data);
        }

        template <class A>
        XSIMD_INLINE batch<float, A> sqrt(batch<float, A> const& self, requires_arch<lsx>) noexcept
        {
            return __lsx_vfsqrt_s(self.data);
        }

        template <class A>
        XSIMD_INLINE batch<double, A> sqrt(batch<double, A> const& self, requires_arch<lsx>) noexcept
        {
            return __lsx_vfsqrt_d(self.data);
        }

        // isnan
        template <class A>
        XSIMD_INLINE batch_bool<float, A> isnan(batch<float, A> const& self, requires_arch<lsx>) noexcept
        {
            return __lsx_vfcmp_cun_s(self.data, self.data);
        }

        template <class A>
        XSIMD_INLINE batch_bool<double, A> isnan(batch<double, A> const& self, requires_arch<lsx>) noexcept
        {
            return __lsx_vfcmp_cun_d(self.data, self.data);
        }

        // select
        template <class A, class T>
        XSIMD_INLINE batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<lsx>) noexcept
        {
            return detail::lsx_from_int<T, A>(__lsx_vbitsel_v(detail::lsx_to_int(false_br), detail::lsx_to_int(true_br), cond.data));
        }

        template <class A, class T, bool... Values>
        XSIMD_INLINE batch<T, A> select(batch_bool_constant<T, A, Values...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<lsx>) noexcept
        {
            return select(batch_bool<T, A> { Values... }, true_br, false_br, lsx {});
        }

        // set
        template <class A, class T, class... Values>
        XSIMD_INLINE batch<T, A> set(batch<T, A> const&, requires_arch<lsx>, Values... values) noexcept
        {
            static_assert(sizeof...(Values) == batch<T, A>::size, "consistent init");
            using vector_type = typename detail::lsx_set_vector<T>::type;
            const vector_type vector = { static_cast<T>(values)... };
            return bit_cast<typename batch<T, A>::register_type>(vector);
        }

        template <class A, class T, class... Values>
        XSIMD_INLINE batch<std::complex<T>, A> set(batch<std::complex<T>, A> const&, requires_arch<lsx>, Values... values) noexcept
        {
            return { set(batch<T, A> {}, lsx {}, values.real()...),
                     set(batch<T, A> {}, lsx {}, values.imag()...) };
        }

        template <class A, class T, class... Values>
        XSIMD_INLINE batch_bool<T, A> set(batch_bool<T, A> const&, requires_arch<lsx>, Values... values) noexcept
        {
            using value_type = sized_uint_t<sizeof(T)>;
            return set(batch<value_type, A> {}, lsx {}, static_cast<value_type>(values ? ~value_type(0) : value_type(0))...).data;
        }

        // byte slides
        template <std::size_t N, class A, class T>
        XSIMD_INLINE batch<T, A> slide_left(batch<T, A> const& self, requires_arch<lsx>) noexcept
        {
            static_assert(N <= 16, "invalid byte slide");
            if constexpr (N == 16)
                return detail::lsx_from_int<T, A>(__lsx_vldi(0));
            else
                return detail::lsx_from_int<T, A>(__lsx_vbsll_v(detail::lsx_to_int(self), N));
        }

        template <std::size_t N, class A, class T>
        XSIMD_INLINE batch<T, A> slide_right(batch<T, A> const& self, requires_arch<lsx>) noexcept
        {
            static_assert(N <= 16, "invalid byte slide");
            if constexpr (N == 16)
                return detail::lsx_from_int<T, A>(__lsx_vldi(0));
            else
                return detail::lsx_from_int<T, A>(__lsx_vbsrl_v(detail::lsx_to_int(self), N));
        }

        // store
        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE void store_unaligned(T* mem, batch<T, A> const& self, requires_arch<lsx>) noexcept
        {
            __lsx_vst(detail::lsx_to_int(self), mem, 0);
        }

        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE void store_aligned(T* mem, batch<T, A> const& self, requires_arch<lsx>) noexcept
        {
            __lsx_vst(detail::lsx_to_int(self), mem, 0);
        }

        // sub
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> sub(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lsx_vsub_b(self.data, other.data);
            else if constexpr (sizeof(T) == 2)
                return __lsx_vsub_h(self.data, other.data);
            else if constexpr (sizeof(T) == 4)
                return __lsx_vsub_w(self.data, other.data);
            else
                return __lsx_vsub_d(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<float, A> sub(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfsub_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<double, A> sub(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lsx>) noexcept
        {
            return __lsx_vfsub_d(self.data, other.data);
        }

        // zip
        template <class A, class T>
        XSIMD_INLINE batch<T, A> zip_lo(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return detail::lsx_from_int<T, A>(__lsx_vilvl_b(detail::lsx_to_int(other), detail::lsx_to_int(self)));
            else if constexpr (sizeof(T) == 2)
                return detail::lsx_from_int<T, A>(__lsx_vilvl_h(detail::lsx_to_int(other), detail::lsx_to_int(self)));
            else if constexpr (sizeof(T) == 4)
                return detail::lsx_from_int<T, A>(__lsx_vilvl_w(detail::lsx_to_int(other), detail::lsx_to_int(self)));
            else
                return detail::lsx_from_int<T, A>(__lsx_vilvl_d(detail::lsx_to_int(other), detail::lsx_to_int(self)));
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> zip_hi(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lsx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return detail::lsx_from_int<T, A>(__lsx_vilvh_b(detail::lsx_to_int(other), detail::lsx_to_int(self)));
            else if constexpr (sizeof(T) == 2)
                return detail::lsx_from_int<T, A>(__lsx_vilvh_h(detail::lsx_to_int(other), detail::lsx_to_int(self)));
            else if constexpr (sizeof(T) == 4)
                return detail::lsx_from_int<T, A>(__lsx_vilvh_w(detail::lsx_to_int(other), detail::lsx_to_int(self)));
            else
                return detail::lsx_from_int<T, A>(__lsx_vilvh_d(detail::lsx_to_int(other), detail::lsx_to_int(self)));
        }
    }
}

#endif
