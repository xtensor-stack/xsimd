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

#ifndef XSIMD_LSX_REGISTER_HPP
#define XSIMD_LSX_REGISTER_HPP

#include "../config/xsimd_config.hpp"
#include "../utils/xsimd_type_traits.hpp"
#include "./xsimd_common_arch.hpp"
#include "./xsimd_register.hpp"

#include <cstddef>

namespace xsimd
{
    struct loongarch : common
    {
    };

    /**
     * @ingroup architectures
     *
     * LoongArch 128-bit SIMD extension.
     */
    struct lsx : loongarch
    {
        static constexpr bool supported() noexcept { return XSIMD_WITH_LSX; }
        static constexpr bool available() noexcept { return true; }
        static constexpr bool requires_alignment() noexcept { return true; }
        static constexpr std::size_t alignment() noexcept { return 16; }
        static constexpr char const* name() noexcept { return "loongarch64+lsx"; }
    };

    /**
     * @ingroup architectures
     *
     * LoongArch 256-bit advanced SIMD extension.
     */
    struct lasx : lsx
    {
        static constexpr bool supported() noexcept { return XSIMD_WITH_LASX; }
        static constexpr bool available() noexcept { return true; }
        static constexpr bool requires_alignment() noexcept { return true; }
        static constexpr std::size_t alignment() noexcept { return 32; }
        static constexpr char const* name() noexcept { return "loongarch64+lasx"; }
    };

#if XSIMD_WITH_LSX || XSIMD_WITH_LASX
    namespace types
    {
        namespace detail
        {
            template <class T, std::size_t Bytes>
            struct loongarch_vector_type;

#define XSIMD_DECLARE_LOONGARCH_VECTOR_TYPE(T, BYTES)       \
    template <>                                             \
    struct loongarch_vector_type<T, BYTES>                  \
    {                                                       \
        typedef T type __attribute__((vector_size(BYTES))); \
    }

#define XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(T) \
    XSIMD_DECLARE_LOONGARCH_VECTOR_TYPE(T, 16); \
    XSIMD_DECLARE_LOONGARCH_VECTOR_TYPE(T, 32)

            XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(signed char);
            XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(unsigned char);
            XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(char);
            XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(short);
            XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(unsigned short);
            XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(int);
            XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(unsigned int);
            XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(long);
            XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(unsigned long);
            XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(long long);
            XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(unsigned long long);
            XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(float);
            XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES(double);

#undef XSIMD_DECLARE_LOONGARCH_VECTOR_TYPES
#undef XSIMD_DECLARE_LOONGARCH_VECTOR_TYPE

            template <class T, class A>
            using loongarch_vector_type_t = typename loongarch_vector_type<T, A::alignment()>::type;

            template <class T>
            using lsx_vector_type_t = loongarch_vector_type_t<T, lsx>;

            template <class T>
            using lasx_vector_type_t = loongarch_vector_type_t<T, lasx>;
        }

#define XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(T, ARCH) \
    XSIMD_DECLARE_SIMD_REGISTER(T, ARCH, detail::ARCH##_vector_type_t<T>)

#define XSIMD_DECLARE_LOONGARCH_SIMD_REGISTERS(ARCH)                 \
    XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(signed char, ARCH);        \
    XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(unsigned char, ARCH);      \
    XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(char, ARCH);               \
    XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(short, ARCH);              \
    XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(unsigned short, ARCH);     \
    XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(int, ARCH);                \
    XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(unsigned int, ARCH);       \
    XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(long, ARCH);               \
    XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(unsigned long, ARCH);      \
    XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(long long, ARCH);          \
    XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(unsigned long long, ARCH); \
    XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(float, ARCH);              \
    XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER(double, ARCH)

#if XSIMD_WITH_LSX
        XSIMD_DECLARE_LOONGARCH_SIMD_REGISTERS(lsx);
#endif
#if XSIMD_WITH_LASX
        XSIMD_DECLARE_LOONGARCH_SIMD_REGISTERS(lasx);
#endif

#undef XSIMD_DECLARE_LOONGARCH_SIMD_REGISTERS
#undef XSIMD_DECLARE_LOONGARCH_SIMD_REGISTER

#if XSIMD_WITH_LSX
        template <class T>
        struct get_bool_simd_register<T, lsx>
        {
            using type = simd_register<xsimd::sized_uint_t<sizeof(T)>, lsx>;
        };
#endif

#if XSIMD_WITH_LASX
        template <class T>
        struct get_bool_simd_register<T, lasx>
        {
            using type = simd_register<xsimd::sized_uint_t<sizeof(T)>, lasx>;
        };
#endif
    }
#endif
}

#endif
