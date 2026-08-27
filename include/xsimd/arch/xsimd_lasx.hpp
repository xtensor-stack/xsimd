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

#ifndef XSIMD_LASX_HPP
#define XSIMD_LASX_HPP

#include "../types/xsimd_batch_constant.hpp"
#include "../types/xsimd_lasx_register.hpp"
#include "../types/xsimd_utils.hpp"
#include "../utils/bits.hpp"

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
            struct lasx_set_vector
            {
                typedef T type __attribute__((vector_size(32)));
            };

            template <class T, class A>
            XSIMD_INLINE __m256i lasx_to_int(batch<T, A> const& value) noexcept
            {
                return bit_cast<__m256i>(value.data);
            }

            template <class T, class A>
            XSIMD_INLINE typename batch<T, A>::register_type lasx_from_int(__m256i value) noexcept
            {
                return bit_cast<typename batch<T, A>::register_type>(value);
            }

            // LASX mask extraction is 128-bit lane-local. Reorder or merge
            // both halves so bit i always represents logical lane i.
            template <class T>
            XSIMD_INLINE std::uint32_t lasx_mask(__m256i value) noexcept
            {
                if constexpr (sizeof(T) == 1)
                {
                    const __m256i sign_bits = __lasx_xvmskltz_b(value);
                    const auto lo = static_cast<std::uint32_t>(__lasx_xvpickve2gr_w(sign_bits, 0));
                    const auto hi = static_cast<std::uint32_t>(__lasx_xvpickve2gr_w(sign_bits, 4));
                    return lo | (hi << 16);
                }
                else if constexpr (sizeof(T) == 2)
                {
                    const __m256i odd = __lasx_xvpickod_b(value, value);
                    const __m256i shuffled = __lasx_xvpermi_d(odd, 0xd8);
                    return static_cast<std::uint32_t>(__lasx_xvpickve2gr_w(__lasx_xvmskltz_b(shuffled), 0));
                }
                else if constexpr (sizeof(T) == 4)
                {
                    const __m256i odd = __lasx_xvpickod_h(value, value);
                    const __m256i shuffled = __lasx_xvpermi_d(odd, 0xd8);
                    return static_cast<std::uint32_t>(__lasx_xvpickve2gr_w(__lasx_xvmskltz_h(shuffled), 0));
                }
                else
                {
                    const __m256i odd = __lasx_xvpickod_w(value, value);
                    const __m256i shuffled = __lasx_xvpermi_d(odd, 0xd8);
                    return static_cast<std::uint32_t>(__lasx_xvpickve2gr_w(__lasx_xvmskltz_w(shuffled), 0));
                }
            }

            XSIMD_INLINE __m256 lasx_hadd_pair(__m256 lhs, __m256 rhs) noexcept
            {
                const __m256i even = __lasx_xvpickev_w(bit_cast<__m256i>(rhs), bit_cast<__m256i>(lhs));
                const __m256i odd = __lasx_xvpickod_w(bit_cast<__m256i>(rhs), bit_cast<__m256i>(lhs));
                return __lasx_xvfadd_s(bit_cast<__m256>(even), bit_cast<__m256>(odd));
            }

            XSIMD_INLINE __m256d lasx_hadd_pair(__m256d lhs, __m256d rhs) noexcept
            {
                const __m256i even = __lasx_xvpickev_d(bit_cast<__m256i>(rhs), bit_cast<__m256i>(lhs));
                const __m256i odd = __lasx_xvpickod_d(bit_cast<__m256i>(rhs), bit_cast<__m256i>(lhs));
                return __lasx_xvfadd_d(bit_cast<__m256d>(even), bit_cast<__m256d>(odd));
            }

            // fast_cast
            template <class A>
            XSIMD_INLINE batch<float, A> fast_cast(batch<std::int32_t, A> const& self, batch<float, A> const&, requires_arch<lasx>) noexcept
            {
                return __lasx_xvffint_s_w(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<float, A> fast_cast(batch<std::uint32_t, A> const& self, batch<float, A> const&, requires_arch<lasx>) noexcept
            {
                return __lasx_xvffint_s_wu(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<double, A> fast_cast(batch<std::int64_t, A> const& self, batch<double, A> const&, requires_arch<lasx>) noexcept
            {
                return __lasx_xvffint_d_l(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<double, A> fast_cast(batch<std::uint64_t, A> const& self, batch<double, A> const&, requires_arch<lasx>) noexcept
            {
                return __lasx_xvffint_d_lu(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<std::int32_t, A> fast_cast(batch<float, A> const& self, batch<std::int32_t, A> const&, requires_arch<lasx>) noexcept
            {
                return __lasx_xvftintrz_w_s(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<std::uint32_t, A> fast_cast(batch<float, A> const& self, batch<std::uint32_t, A> const&, requires_arch<lasx>) noexcept
            {
                return __lasx_xvftintrz_wu_s(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<std::int64_t, A> fast_cast(batch<double, A> const& self, batch<std::int64_t, A> const&, requires_arch<lasx>) noexcept
            {
                return __lasx_xvftintrz_l_d(self.data);
            }

            template <class A>
            XSIMD_INLINE batch<std::uint64_t, A> fast_cast(batch<double, A> const& self, batch<std::uint64_t, A> const&, requires_arch<lasx>) noexcept
            {
                return __lasx_xvftintrz_lu_d(self.data);
            }
        }

        // abs
        template <class A, class T, class = std::enable_if_t<std::is_floating_point_v<T>>>
        XSIMD_INLINE batch<T, A> abs(batch<T, A> const& self, requires_arch<lasx>) noexcept
        {
            if constexpr (sizeof(T) == 4)
                return detail::lasx_from_int<T, A>(__lasx_xvbitclri_w(detail::lasx_to_int(self), 8 * sizeof(T) - 1));
            else
                return detail::lasx_from_int<T, A>(__lasx_xvbitclri_d(detail::lasx_to_int(self), 8 * sizeof(T) - 1));
        }

        // add
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> add(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lasx_xvadd_b(self.data, other.data);
            else if constexpr (sizeof(T) == 2)
                return __lasx_xvadd_h(self.data, other.data);
            else if constexpr (sizeof(T) == 4)
                return __lasx_xvadd_w(self.data, other.data);
            else
                return __lasx_xvadd_d(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<float, A> add(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfadd_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<double, A> add(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfadd_d(self.data, other.data);
        }

        // all/any
        template <class A, class T>
        XSIMD_INLINE bool all(batch_bool<T, A> const& self, requires_arch<lasx>) noexcept
        {
            constexpr std::uint64_t all_bits = utils::make_low_mask(static_cast<std::uint64_t>(batch_bool<T, A>::size));
            return detail::lasx_mask<T>(self.data) == all_bits;
        }

        template <class A, class T>
        XSIMD_INLINE bool any(batch_bool<T, A> const& self, requires_arch<lasx>) noexcept
        {
            return detail::lasx_mask<T>(self.data) != 0;
        }

        // batch_bool_cast
        template <class A, class T_out, class T_in>
        XSIMD_INLINE batch_bool<T_out, A> batch_bool_cast(batch_bool<T_in, A> const& self, batch_bool<T_out, A> const&, requires_arch<lasx>) noexcept
        {
            return self.data;
        }

        // bitwise operations
        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_and(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            return detail::lasx_from_int<T, A>(__lasx_xvand_v(detail::lasx_to_int(self), detail::lasx_to_int(other)));
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_and(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvand_v(self.data, other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_andnot(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            return detail::lasx_from_int<T, A>(__lasx_xvandn_v(detail::lasx_to_int(other), detail::lasx_to_int(self)));
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_andnot(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvandn_v(other.data, self.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_not(batch<T, A> const& self, requires_arch<lasx>) noexcept
        {
            const __m256i bits = detail::lasx_to_int(self);
            return detail::lasx_from_int<T, A>(__lasx_xvnor_v(bits, bits));
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_not(batch_bool<T, A> const& self, requires_arch<lasx>) noexcept
        {
            return __lasx_xvnor_v(self.data, self.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_or(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            return detail::lasx_from_int<T, A>(__lasx_xvor_v(detail::lasx_to_int(self), detail::lasx_to_int(other)));
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_or(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvor_v(self.data, other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_xor(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            return detail::lasx_from_int<T, A>(__lasx_xvxor_v(detail::lasx_to_int(self), detail::lasx_to_int(other)));
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_xor(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvxor_v(self.data, other.data);
        }

        template <class A, class T_in, class T_out>
        XSIMD_INLINE batch<T_out, A> bitwise_cast(batch<T_in, A> const& self, batch<T_out, A> const&, requires_arch<lasx>) noexcept
        {
            return bit_cast<typename batch<T_out, A>::register_type>(self.data);
        }

        // shifts
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> bitwise_lshift(batch<T, A> const& self, std::int32_t other, requires_arch<lasx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lasx_xvsll_b(self.data, __lasx_xvreplgr2vr_b(other));
            else if constexpr (sizeof(T) == 2)
                return __lasx_xvsll_h(self.data, __lasx_xvreplgr2vr_h(other));
            else if constexpr (sizeof(T) == 4)
                return __lasx_xvsll_w(self.data, __lasx_xvreplgr2vr_w(other));
            else
                return __lasx_xvsll_d(self.data, __lasx_xvreplgr2vr_d(other));
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> bitwise_rshift(batch<T, A> const& self, std::int32_t other, requires_arch<lasx>) noexcept
        {
            if constexpr (std::is_signed_v<T>)
            {
                if constexpr (sizeof(T) == 1)
                    return __lasx_xvsra_b(self.data, __lasx_xvreplgr2vr_b(other));
                else if constexpr (sizeof(T) == 2)
                    return __lasx_xvsra_h(self.data, __lasx_xvreplgr2vr_h(other));
                else if constexpr (sizeof(T) == 4)
                    return __lasx_xvsra_w(self.data, __lasx_xvreplgr2vr_w(other));
                else
                    return __lasx_xvsra_d(self.data, __lasx_xvreplgr2vr_d(other));
            }
            else
            {
                if constexpr (sizeof(T) == 1)
                    return __lasx_xvsrl_b(self.data, __lasx_xvreplgr2vr_b(other));
                else if constexpr (sizeof(T) == 2)
                    return __lasx_xvsrl_h(self.data, __lasx_xvreplgr2vr_h(other));
                else if constexpr (sizeof(T) == 4)
                    return __lasx_xvsrl_w(self.data, __lasx_xvreplgr2vr_w(other));
                else
                    return __lasx_xvsrl_d(self.data, __lasx_xvreplgr2vr_d(other));
            }
        }

        // div
        template <class A>
        XSIMD_INLINE batch<float, A> div(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfdiv_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<double, A> div(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfdiv_d(self.data, other.data);
        }

        // broadcast
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> broadcast(T value, requires_arch<lasx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lasx_xvreplgr2vr_b(static_cast<int>(value));
            else if constexpr (sizeof(T) == 2)
                return __lasx_xvreplgr2vr_h(static_cast<int>(value));
            else if constexpr (sizeof(T) == 4)
                return __lasx_xvreplgr2vr_w(bit_cast<sized_int_t<sizeof(T)>>(value));
            else
                return __lasx_xvreplgr2vr_d(bit_cast<sized_int_t<sizeof(T)>>(value));
        }

        template <class A>
        XSIMD_INLINE batch<float, A> broadcast(float value, requires_arch<lasx>) noexcept
        {
            const auto bits = bit_cast<std::int32_t>(value);
            return bit_cast<__m256>(__lasx_xvreplgr2vr_w(bits));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> broadcast(double value, requires_arch<lasx>) noexcept
        {
            const auto bits = bit_cast<std::int64_t>(value);
            return bit_cast<__m256d>(__lasx_xvreplgr2vr_d(static_cast<long>(bits)));
        }

        // comparisons
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lasx_xvseq_b(self.data, other.data);
            else if constexpr (sizeof(T) == 2)
                return __lasx_xvseq_h(self.data, other.data);
            else if constexpr (sizeof(T) == 4)
                return __lasx_xvseq_w(self.data, other.data);
            else
                return __lasx_xvseq_d(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch_bool<float, A> eq(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfcmp_ceq_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch_bool<double, A> eq(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfcmp_ceq_d(self.data, other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> eq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<lasx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lasx_xvseq_b(self.data, other.data);
            else if constexpr (sizeof(T) == 2)
                return __lasx_xvseq_h(self.data, other.data);
            else if constexpr (sizeof(T) == 4)
                return __lasx_xvseq_w(self.data, other.data);
            else
                return __lasx_xvseq_d(self.data, other.data);
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch_bool<T, A> lt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            if constexpr (std::is_signed_v<T>)
            {
                if constexpr (sizeof(T) == 1)
                    return __lasx_xvslt_b(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lasx_xvslt_h(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lasx_xvslt_w(self.data, other.data);
                else
                    return __lasx_xvslt_d(self.data, other.data);
            }
            else
            {
                if constexpr (sizeof(T) == 1)
                    return __lasx_xvslt_bu(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lasx_xvslt_hu(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lasx_xvslt_wu(self.data, other.data);
                else
                    return __lasx_xvslt_du(self.data, other.data);
            }
        }

        template <class A>
        XSIMD_INLINE batch_bool<float, A> lt(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfcmp_clt_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch_bool<double, A> lt(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfcmp_clt_d(self.data, other.data);
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch_bool<T, A> le(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            if constexpr (std::is_signed_v<T>)
            {
                if constexpr (sizeof(T) == 1)
                    return __lasx_xvsle_b(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lasx_xvsle_h(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lasx_xvsle_w(self.data, other.data);
                else
                    return __lasx_xvsle_d(self.data, other.data);
            }
            else
            {
                if constexpr (sizeof(T) == 1)
                    return __lasx_xvsle_bu(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lasx_xvsle_hu(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lasx_xvsle_wu(self.data, other.data);
                else
                    return __lasx_xvsle_du(self.data, other.data);
            }
        }

        template <class A>
        XSIMD_INLINE batch_bool<float, A> le(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfcmp_cle_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch_bool<double, A> le(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfcmp_cle_d(self.data, other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            return lt(other, self, lasx {});
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> ge(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            return le(other, self, lasx {});
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            const __m256i equal = eq(self, other, lasx {}).data;
            return __lasx_xvnor_v(equal, equal);
        }

        template <class A>
        XSIMD_INLINE batch_bool<float, A> neq(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfcmp_cune_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch_bool<double, A> neq(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfcmp_cune_d(self.data, other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> neq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvxor_v(self.data, other.data);
        }

        // first
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE T first(batch<T, A> const& self, requires_arch<lasx>) noexcept
        {
            if constexpr (sizeof(T) <= 4)
            {
                const auto word = static_cast<std::uint32_t>(__lasx_xvpickve2gr_wu(self.data, 0));
                using unsigned_type = sized_uint_t<sizeof(T)>;
                return bit_cast<T>(static_cast<unsigned_type>(word));
            }
            else if constexpr (std::is_signed_v<T>)
                return static_cast<T>(__lasx_xvpickve2gr_d(self.data, 0));
            else
                return static_cast<T>(__lasx_xvpickve2gr_du(self.data, 0));
        }

        template <class A>
        XSIMD_INLINE float first(batch<float, A> const& self, requires_arch<lasx>) noexcept
        {
            const auto bits = static_cast<std::uint32_t>(__lasx_xvpickve2gr_w(detail::lasx_to_int(self), 0));
            return bit_cast<float>(bits);
        }

        template <class A>
        XSIMD_INLINE double first(batch<double, A> const& self, requires_arch<lasx>) noexcept
        {
            const auto bits = static_cast<std::uint64_t>(__lasx_xvpickve2gr_du(detail::lasx_to_int(self), 0));
            return bit_cast<double>(bits);
        }

        template <class A, class T>
        XSIMD_INLINE std::complex<T> first(batch<std::complex<T>, A> const& self, requires_arch<lasx>) noexcept
        {
            return { first(self.real(), lasx {}), first(self.imag(), lasx {}) };
        }

        // horizontal add of rows
        template <class A>
        XSIMD_INLINE batch<float, A> haddp(batch<float, A> const* row, requires_arch<lasx>) noexcept
        {
            __m256 abcd = detail::lasx_hadd_pair(
                detail::lasx_hadd_pair(row[0].data, row[1].data),
                detail::lasx_hadd_pair(row[2].data, row[3].data));
            __m256 efgh = detail::lasx_hadd_pair(
                detail::lasx_hadd_pair(row[4].data, row[5].data),
                detail::lasx_hadd_pair(row[6].data, row[7].data));
            const __m256i lo = __lasx_xvpermi_q(bit_cast<__m256i>(efgh), bit_cast<__m256i>(abcd), 0x30);
            const __m256i hi = __lasx_xvpermi_q(bit_cast<__m256i>(efgh), bit_cast<__m256i>(abcd), 0x21);
            return __lasx_xvfadd_s(bit_cast<__m256>(lo), bit_cast<__m256>(hi));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> haddp(batch<double, A> const* row, requires_arch<lasx>) noexcept
        {
            const __m256d ab = detail::lasx_hadd_pair(row[0].data, row[1].data);
            const __m256d cd = detail::lasx_hadd_pair(row[2].data, row[3].data);
            const __m256i lo = __lasx_xvpermi_q(bit_cast<__m256i>(cd), bit_cast<__m256i>(ab), 0x30);
            const __m256i hi = __lasx_xvpermi_q(bit_cast<__m256i>(cd), bit_cast<__m256i>(ab), 0x21);
            return __lasx_xvfadd_d(bit_cast<__m256d>(lo), bit_cast<__m256d>(hi));
        }

        // load
        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE batch<T, A> load_unaligned(T const* mem, convert<T>, requires_arch<lasx>) noexcept
        {
            return detail::lasx_from_int<T, A>(__lasx_xvld(mem, 0));
        }

        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE batch<T, A> load_aligned(T const* mem, convert<T>, requires_arch<lasx>) noexcept
        {
            return detail::lasx_from_int<T, A>(__lasx_xvld(mem, 0));
        }

        // load/store complex helpers
        namespace detail
        {
            template <class A, class T>
            XSIMD_INLINE batch<std::complex<T>, A> load_complex(batch<T, A> const& first_chunk, batch<T, A> const& second_chunk, requires_arch<lasx>) noexcept
            {
                __m256i real;
                __m256i imag;
                if constexpr (sizeof(T) == 4)
                {
                    real = __lasx_xvpickev_w(lasx_to_int(second_chunk), lasx_to_int(first_chunk));
                    imag = __lasx_xvpickod_w(lasx_to_int(second_chunk), lasx_to_int(first_chunk));
                }
                else
                {
                    real = __lasx_xvpickev_d(lasx_to_int(second_chunk), lasx_to_int(first_chunk));
                    imag = __lasx_xvpickod_d(lasx_to_int(second_chunk), lasx_to_int(first_chunk));
                }
                // xvpick* operates independently on each 128-bit lane; restore
                // the linear element order expected by batch<complex<T>>.
                real = __lasx_xvpermi_d(real, 0xd8);
                imag = __lasx_xvpermi_d(imag, 0xd8);
                return { lasx_from_int<T, A>(real), lasx_from_int<T, A>(imag) };
            }

            template <class A, class T>
            XSIMD_INLINE batch<T, A> complex_low(batch<std::complex<T>, A> const& self, requires_arch<lasx>) noexcept
            {
                __m256i lo;
                __m256i hi;
                if constexpr (sizeof(T) == 4)
                {
                    lo = __lasx_xvilvl_w(lasx_to_int(self.imag()), lasx_to_int(self.real()));
                    hi = __lasx_xvilvh_w(lasx_to_int(self.imag()), lasx_to_int(self.real()));
                }
                else
                {
                    lo = __lasx_xvilvl_d(lasx_to_int(self.imag()), lasx_to_int(self.real()));
                    hi = __lasx_xvilvh_d(lasx_to_int(self.imag()), lasx_to_int(self.real()));
                }
                return lasx_from_int<T, A>(__lasx_xvpermi_q(hi, lo, 0x20));
            }

            template <class A, class T>
            XSIMD_INLINE batch<T, A> complex_high(batch<std::complex<T>, A> const& self, requires_arch<lasx>) noexcept
            {
                __m256i lo;
                __m256i hi;
                if constexpr (sizeof(T) == 4)
                {
                    lo = __lasx_xvilvl_w(lasx_to_int(self.imag()), lasx_to_int(self.real()));
                    hi = __lasx_xvilvh_w(lasx_to_int(self.imag()), lasx_to_int(self.real()));
                }
                else
                {
                    lo = __lasx_xvilvl_d(lasx_to_int(self.imag()), lasx_to_int(self.real()));
                    hi = __lasx_xvilvh_d(lasx_to_int(self.imag()), lasx_to_int(self.real()));
                }
                return lasx_from_int<T, A>(__lasx_xvpermi_q(hi, lo, 0x31));
            }
        }

        // max/min
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            if constexpr (std::is_signed_v<T>)
            {
                if constexpr (sizeof(T) == 1)
                    return __lasx_xvmax_b(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lasx_xvmax_h(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lasx_xvmax_w(self.data, other.data);
                else
                    return __lasx_xvmax_d(self.data, other.data);
            }
            else
            {
                if constexpr (sizeof(T) == 1)
                    return __lasx_xvmax_bu(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lasx_xvmax_hu(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lasx_xvmax_wu(self.data, other.data);
                else
                    return __lasx_xvmax_du(self.data, other.data);
            }
        }

        template <class A>
        XSIMD_INLINE batch<float, A> max(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lasx>) noexcept
        {
            const __m256i cond = __lasx_xvfcmp_clt_s(self.data, other.data);
            return detail::lasx_from_int<float, A>(__lasx_xvbitsel_v(detail::lasx_to_int(self), detail::lasx_to_int(other), cond));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> max(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lasx>) noexcept
        {
            const __m256i cond = __lasx_xvfcmp_clt_d(self.data, other.data);
            return detail::lasx_from_int<double, A>(__lasx_xvbitsel_v(detail::lasx_to_int(self), detail::lasx_to_int(other), cond));
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            if constexpr (std::is_signed_v<T>)
            {
                if constexpr (sizeof(T) == 1)
                    return __lasx_xvmin_b(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lasx_xvmin_h(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lasx_xvmin_w(self.data, other.data);
                else
                    return __lasx_xvmin_d(self.data, other.data);
            }
            else
            {
                if constexpr (sizeof(T) == 1)
                    return __lasx_xvmin_bu(self.data, other.data);
                else if constexpr (sizeof(T) == 2)
                    return __lasx_xvmin_hu(self.data, other.data);
                else if constexpr (sizeof(T) == 4)
                    return __lasx_xvmin_wu(self.data, other.data);
                else
                    return __lasx_xvmin_du(self.data, other.data);
            }
        }

        template <class A>
        XSIMD_INLINE batch<float, A> min(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lasx>) noexcept
        {
            const __m256i cond = __lasx_xvfcmp_clt_s(other.data, self.data);
            return detail::lasx_from_int<float, A>(__lasx_xvbitsel_v(detail::lasx_to_int(self), detail::lasx_to_int(other), cond));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> min(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lasx>) noexcept
        {
            const __m256i cond = __lasx_xvfcmp_clt_d(other.data, self.data);
            return detail::lasx_from_int<double, A>(__lasx_xvbitsel_v(detail::lasx_to_int(self), detail::lasx_to_int(other), cond));
        }

        // mul/neg
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lasx_xvmul_b(self.data, other.data);
            else if constexpr (sizeof(T) == 2)
                return __lasx_xvmul_h(self.data, other.data);
            else if constexpr (sizeof(T) == 4)
                return __lasx_xvmul_w(self.data, other.data);
            else
                return __lasx_xvmul_d(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<float, A> mul(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfmul_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<double, A> mul(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfmul_d(self.data, other.data);
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> neg(batch<T, A> const& self, requires_arch<lasx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lasx_xvneg_b(self.data);
            else if constexpr (sizeof(T) == 2)
                return __lasx_xvneg_h(self.data);
            else if constexpr (sizeof(T) == 4)
                return __lasx_xvneg_w(self.data);
            else
                return __lasx_xvneg_d(self.data);
        }

        template <class A>
        XSIMD_INLINE batch<float, A> neg(batch<float, A> const& self, requires_arch<lasx>) noexcept
        {
            const auto sign_bit = bit_cast<std::int32_t>(std::uint32_t(1) << 31);
            const __m256i sign = __lasx_xvreplgr2vr_w(sign_bit);
            return detail::lasx_from_int<float, A>(__lasx_xvxor_v(detail::lasx_to_int(self), sign));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> neg(batch<double, A> const& self, requires_arch<lasx>) noexcept
        {
            const auto sign_bit = bit_cast<std::int64_t>(std::uint64_t(1) << 63);
            const __m256i sign = __lasx_xvreplgr2vr_d(sign_bit);
            return detail::lasx_from_int<double, A>(__lasx_xvxor_v(detail::lasx_to_int(self), sign));
        }

        // rsqrt/sqrt
        template <class A>
        XSIMD_INLINE batch<float, A> rsqrt(batch<float, A> const& self, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfrsqrt_s(self.data);
        }

        template <class A>
        XSIMD_INLINE batch<double, A> rsqrt(batch<double, A> const& self, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfrsqrt_d(self.data);
        }

        template <class A>
        XSIMD_INLINE batch<float, A> sqrt(batch<float, A> const& self, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfsqrt_s(self.data);
        }

        template <class A>
        XSIMD_INLINE batch<double, A> sqrt(batch<double, A> const& self, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfsqrt_d(self.data);
        }

        // isnan
        template <class A>
        XSIMD_INLINE batch_bool<float, A> isnan(batch<float, A> const& self, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfcmp_cun_s(self.data, self.data);
        }

        template <class A>
        XSIMD_INLINE batch_bool<double, A> isnan(batch<double, A> const& self, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfcmp_cun_d(self.data, self.data);
        }

        // select
        template <class A, class T>
        XSIMD_INLINE batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<lasx>) noexcept
        {
            return detail::lasx_from_int<T, A>(__lasx_xvbitsel_v(detail::lasx_to_int(false_br), detail::lasx_to_int(true_br), cond.data));
        }

        template <class A, class T, bool... Values>
        XSIMD_INLINE batch<T, A> select(batch_bool_constant<T, A, Values...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<lasx>) noexcept
        {
            return select(batch_bool<T, A> { Values... }, true_br, false_br, lasx {});
        }

        // set
        template <class A, class T, class... Values>
        XSIMD_INLINE batch<T, A> set(batch<T, A> const&, requires_arch<lasx>, Values... values) noexcept
        {
            static_assert(sizeof...(Values) == batch<T, A>::size, "consistent init");
            using vector_type = typename detail::lasx_set_vector<T>::type;
            const vector_type vector = { static_cast<T>(values)... };
            return bit_cast<typename batch<T, A>::register_type>(vector);
        }

        template <class A, class T, class... Values>
        XSIMD_INLINE batch<std::complex<T>, A> set(batch<std::complex<T>, A> const&, requires_arch<lasx>, Values... values) noexcept
        {
            return { set(batch<T, A> {}, lasx {}, values.real()...),
                     set(batch<T, A> {}, lasx {}, values.imag()...) };
        }

        template <class A, class T, class... Values>
        XSIMD_INLINE batch_bool<T, A> set(batch_bool<T, A> const&, requires_arch<lasx>, Values... values) noexcept
        {
            using value_type = sized_uint_t<sizeof(T)>;
            return set(batch<value_type, A> {}, lasx {}, static_cast<value_type>(values ? ~value_type(0) : value_type(0))...).data;
        }

        // byte slides
        // xvbsll/xvbsrl operate independently on each 128-bit lane, so the
        // adjacent lane is permuted in to supply or receive the carried bytes.
        template <std::size_t N, class A, class T>
        XSIMD_INLINE batch<T, A> slide_left(batch<T, A> const& self, requires_arch<lasx>) noexcept
        {
            static_assert(N <= 32, "invalid byte slide");
            const __m256i bits = detail::lasx_to_int(self);
            const __m256i zero = __lasx_xvldi(0);
            if constexpr (N == 0)
                return self;
            else if constexpr (N < 16)
            {
                const __m256i shifted = __lasx_xvbsll_v(bits, N);
                const __m256i previous = __lasx_xvpermi_q(bits, zero, 0x20);
                const __m256i carry = __lasx_xvbsrl_v(previous, 16 - N);
                return detail::lasx_from_int<T, A>(__lasx_xvor_v(shifted, carry));
            }
            else if constexpr (N == 16)
                return detail::lasx_from_int<T, A>(__lasx_xvpermi_q(bits, zero, 0x20));
            else if constexpr (N < 32)
            {
                const __m256i previous = __lasx_xvpermi_q(bits, zero, 0x20);
                return detail::lasx_from_int<T, A>(__lasx_xvbsll_v(previous, N - 16));
            }
            else
                return detail::lasx_from_int<T, A>(zero);
        }

        template <std::size_t N, class A, class T>
        XSIMD_INLINE batch<T, A> slide_right(batch<T, A> const& self, requires_arch<lasx>) noexcept
        {
            static_assert(N <= 32, "invalid byte slide");
            const __m256i bits = detail::lasx_to_int(self);
            const __m256i zero = __lasx_xvldi(0);
            if constexpr (N == 0)
                return self;
            else if constexpr (N < 16)
            {
                const __m256i shifted = __lasx_xvbsrl_v(bits, N);
                const __m256i next = __lasx_xvpermi_q(zero, bits, 0x31);
                const __m256i carry = __lasx_xvbsll_v(next, 16 - N);
                return detail::lasx_from_int<T, A>(__lasx_xvor_v(shifted, carry));
            }
            else if constexpr (N == 16)
                return detail::lasx_from_int<T, A>(__lasx_xvpermi_q(zero, bits, 0x31));
            else if constexpr (N < 32)
            {
                const __m256i next = __lasx_xvpermi_q(zero, bits, 0x31);
                return detail::lasx_from_int<T, A>(__lasx_xvbsrl_v(next, N - 16));
            }
            else
                return detail::lasx_from_int<T, A>(zero);
        }

        // store
        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE void store_unaligned(T* mem, batch<T, A> const& self, requires_arch<lasx>) noexcept
        {
            __lasx_xvst(detail::lasx_to_int(self), mem, 0);
        }

        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE void store_aligned(T* mem, batch<T, A> const& self, requires_arch<lasx>) noexcept
        {
            __lasx_xvst(detail::lasx_to_int(self), mem, 0);
        }

        // sub
        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> sub(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            if constexpr (sizeof(T) == 1)
                return __lasx_xvsub_b(self.data, other.data);
            else if constexpr (sizeof(T) == 2)
                return __lasx_xvsub_h(self.data, other.data);
            else if constexpr (sizeof(T) == 4)
                return __lasx_xvsub_w(self.data, other.data);
            else
                return __lasx_xvsub_d(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<float, A> sub(batch<float, A> const& self, batch<float, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfsub_s(self.data, other.data);
        }

        template <class A>
        XSIMD_INLINE batch<double, A> sub(batch<double, A> const& self, batch<double, A> const& other, requires_arch<lasx>) noexcept
        {
            return __lasx_xvfsub_d(self.data, other.data);
        }

        // zip
        // xvilvl/xvilvh interleave within 128-bit lanes; xvpermi_q assembles
        // the full-width lower or upper half of the logical 256-bit vectors.
        template <class A, class T>
        XSIMD_INLINE batch<T, A> zip_lo(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            __m256i lo;
            __m256i hi;
            if constexpr (sizeof(T) == 1)
            {
                lo = __lasx_xvilvl_b(detail::lasx_to_int(other), detail::lasx_to_int(self));
                hi = __lasx_xvilvh_b(detail::lasx_to_int(other), detail::lasx_to_int(self));
            }
            else if constexpr (sizeof(T) == 2)
            {
                lo = __lasx_xvilvl_h(detail::lasx_to_int(other), detail::lasx_to_int(self));
                hi = __lasx_xvilvh_h(detail::lasx_to_int(other), detail::lasx_to_int(self));
            }
            else if constexpr (sizeof(T) == 4)
            {
                lo = __lasx_xvilvl_w(detail::lasx_to_int(other), detail::lasx_to_int(self));
                hi = __lasx_xvilvh_w(detail::lasx_to_int(other), detail::lasx_to_int(self));
            }
            else
            {
                lo = __lasx_xvilvl_d(detail::lasx_to_int(other), detail::lasx_to_int(self));
                hi = __lasx_xvilvh_d(detail::lasx_to_int(other), detail::lasx_to_int(self));
            }
            return detail::lasx_from_int<T, A>(__lasx_xvpermi_q(hi, lo, 0x20));
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> zip_hi(batch<T, A> const& self, batch<T, A> const& other, requires_arch<lasx>) noexcept
        {
            __m256i lo;
            __m256i hi;
            if constexpr (sizeof(T) == 1)
            {
                lo = __lasx_xvilvl_b(detail::lasx_to_int(other), detail::lasx_to_int(self));
                hi = __lasx_xvilvh_b(detail::lasx_to_int(other), detail::lasx_to_int(self));
            }
            else if constexpr (sizeof(T) == 2)
            {
                lo = __lasx_xvilvl_h(detail::lasx_to_int(other), detail::lasx_to_int(self));
                hi = __lasx_xvilvh_h(detail::lasx_to_int(other), detail::lasx_to_int(self));
            }
            else if constexpr (sizeof(T) == 4)
            {
                lo = __lasx_xvilvl_w(detail::lasx_to_int(other), detail::lasx_to_int(self));
                hi = __lasx_xvilvh_w(detail::lasx_to_int(other), detail::lasx_to_int(self));
            }
            else
            {
                lo = __lasx_xvilvl_d(detail::lasx_to_int(other), detail::lasx_to_int(self));
                hi = __lasx_xvilvh_d(detail::lasx_to_int(other), detail::lasx_to_int(self));
            }
            return detail::lasx_from_int<T, A>(__lasx_xvpermi_q(hi, lo, 0x31));
        }
    }
}

#endif
