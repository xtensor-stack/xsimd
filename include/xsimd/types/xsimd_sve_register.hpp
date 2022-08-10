/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 * Copyright (c) Yibo Cai                                                   *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_SVE_REGISTER_HPP
#define XSIMD_SVE_REGISTER_HPP

#include "xsimd_generic_arch.hpp"
#include "xsimd_register.hpp"

#if XSIMD_WITH_SVE
#include <arm_sve.h>
#endif

namespace xsimd
{
    /**
     * @ingroup arch
     *
     * SVE instructions (fixed vector size) for arm64
     */
    namespace detail
    {
        struct sve : xsimd::generic
        {
            static constexpr bool supported() noexcept { return XSIMD_WITH_SVE; }
            static constexpr bool available() noexcept { return true; }
            static constexpr bool requires_alignment() noexcept { return true; }
            static constexpr std::size_t alignment() noexcept { return 16; }
            static constexpr unsigned version() noexcept { return generic::version(9, 0, 0); }
            static constexpr char const* name() noexcept { return "arm64+sve"; }
        };
    } // namespace detail

#if XSIMD_WITH_SVE

    // Create alias with explicit vector size
#if XSIMD_SVE_BITS == 128
    using sve128 = detail::sve;
#elif XSIMD_SVE_BITS == 256
    using sve256 = detail::sve;
#endif

    namespace types
    {
        namespace detail
        {
// define fixed size alias per SVE sizeless type
#define SVE_TO_FIXED_SIZE(ty) ty __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)))
            using sve_int8_t = SVE_TO_FIXED_SIZE(svint8_t);
            using sve_uint8_t = SVE_TO_FIXED_SIZE(svuint8_t);
            using sve_int16_t = SVE_TO_FIXED_SIZE(svint16_t);
            using sve_uint16_t = SVE_TO_FIXED_SIZE(svuint16_t);
            using sve_int32_t = SVE_TO_FIXED_SIZE(svint32_t);
            using sve_uint32_t = SVE_TO_FIXED_SIZE(svuint32_t);
            using sve_int64_t = SVE_TO_FIXED_SIZE(svint64_t);
            using sve_uint64_t = SVE_TO_FIXED_SIZE(svuint64_t);
            using sve_float32_t = SVE_TO_FIXED_SIZE(svfloat32_t);
            using sve_float64_t = SVE_TO_FIXED_SIZE(svfloat64_t);
            using sve_bool_t = SVE_TO_FIXED_SIZE(svbool_t);
#undef SVE_TO_FIXED_SIZE

            template <size_t S>
            struct sve_vector_type_impl;

            template <>
            struct sve_vector_type_impl<8>
            {
                using signed_type = sve_int8_t;
                using unsigned_type = sve_uint8_t;
                using floating_point_type = void;
            };

            template <>
            struct sve_vector_type_impl<16>
            {
                using signed_type = sve_int16_t;
                using unsigned_type = sve_uint16_t;
                using floating_point_type = void;
            };

            template <>
            struct sve_vector_type_impl<32>
            {
                using signed_type = sve_int32_t;
                using unsigned_type = sve_uint32_t;
                using floating_point_type = sve_float32_t;
            };

            template <>
            struct sve_vector_type_impl<64>
            {
                using signed_type = sve_int64_t;
                using unsigned_type = sve_uint64_t;
                using floating_point_type = sve_float64_t;
            };

            template <class T>
            using signed_int_sve_vector_type = typename sve_vector_type_impl<8 * sizeof(T)>::signed_type;

            template <class T>
            using unsigned_int_sve_vector_type = typename sve_vector_type_impl<8 * sizeof(T)>::unsigned_type;

            template <class T>
            using floating_point_sve_vector_type = typename sve_vector_type_impl<8 * sizeof(T)>::floating_point_type;

            template <class T>
            using signed_int_or_floating_point_sve_vector_type = typename std::conditional<std::is_floating_point<T>::value,
                                                                                           floating_point_sve_vector_type<T>,
                                                                                           signed_int_sve_vector_type<T>>::type;

            template <class T>
            using sve_vector_type = typename std::conditional<std::is_signed<T>::value,
                                                              signed_int_or_floating_point_sve_vector_type<T>,
                                                              unsigned_int_sve_vector_type<T>>::type;
        } // namespace detail

        XSIMD_DECLARE_SIMD_REGISTER(signed char, xsimd::detail::sve, detail::sve_vector_type<signed char>);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned char, xsimd::detail::sve, detail::sve_vector_type<unsigned char>);
        XSIMD_DECLARE_SIMD_REGISTER(char, xsimd::detail::sve, detail::sve_vector_type<char>);
        XSIMD_DECLARE_SIMD_REGISTER(short, xsimd::detail::sve, detail::sve_vector_type<short>);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned short, xsimd::detail::sve, detail::sve_vector_type<unsigned short>);
        XSIMD_DECLARE_SIMD_REGISTER(int, xsimd::detail::sve, detail::sve_vector_type<int>);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned int, xsimd::detail::sve, detail::sve_vector_type<unsigned int>);
        XSIMD_DECLARE_SIMD_REGISTER(long int, xsimd::detail::sve, detail::sve_vector_type<long int>);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned long int, xsimd::detail::sve, detail::sve_vector_type<unsigned long int>);
        XSIMD_DECLARE_SIMD_REGISTER(long long int, xsimd::detail::sve, detail::sve_vector_type<long long int>);
        XSIMD_DECLARE_SIMD_REGISTER(unsigned long long int, xsimd::detail::sve, detail::sve_vector_type<unsigned long long int>);
        XSIMD_DECLARE_SIMD_REGISTER(float, xsimd::detail::sve, detail::sve_vector_type<float>);
        XSIMD_DECLARE_SIMD_REGISTER(double, xsimd::detail::sve, detail::sve_vector_type<double>);

        namespace detail
        {
            struct sve_bool_simd_register
            {
                using register_type = sve_bool_t;
                register_type data;
                operator register_type() const noexcept { return data; }
            };
        } // namespace detail

        template <class T>
        struct get_bool_simd_register<T, xsimd::detail::sve>
        {
            using type = detail::sve_bool_simd_register;
        };
    } // namespace types
#endif
} // namespace xsimd

#endif
