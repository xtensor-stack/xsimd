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

#include <array>
#include <cmath>
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
            template <class To, class From>
            XSIMD_INLINE To loongarch_bit_cast(From const& value) noexcept
            {
                static_assert(sizeof(To) == sizeof(From), "incompatible vector sizes");
                To result;
                __builtin_memcpy(&result, &value, sizeof(result));
                return result;
            }

            template <class T, class A>
            using loongarch_unsigned_register_t = typename batch<as_unsigned_integer_t<T>, A>::register_type;

            template <class T, class A>
            XSIMD_INLINE loongarch_unsigned_register_t<T, A> loongarch_to_bits(batch<T, A> const& value) noexcept
            {
                return loongarch_bit_cast<loongarch_unsigned_register_t<T, A>>(value.data);
            }

            template <class T, class A>
            XSIMD_INLINE typename batch<T, A>::register_type loongarch_from_bits(loongarch_unsigned_register_t<T, A> const& value) noexcept
            {
                return loongarch_bit_cast<typename batch<T, A>::register_type>(value);
            }

            template <class A>
            XSIMD_INLINE batch<float, A> fast_cast(batch<std::int32_t, A> const& self, batch<float, A> const&, requires_arch<loongarch>) noexcept
            {
                typename batch<float, A>::register_type result {};
                for (std::size_t i = 0; i < batch<float, A>::size; ++i)
                {
                    result[i] = static_cast<float>(self.data[i]);
                }
                return result;
            }

            template <class A>
            XSIMD_INLINE batch<std::int32_t, A> fast_cast(batch<float, A> const& self, batch<std::int32_t, A> const&, requires_arch<loongarch>) noexcept
            {
                typename batch<std::int32_t, A>::register_type result {};
                for (std::size_t i = 0; i < batch<float, A>::size; ++i)
                {
                    result[i] = static_cast<std::int32_t>(self.data[i]);
                }
                return result;
            }
        }

        // abs
        template <class A, class T, class = std::enable_if_t<std::is_floating_point_v<T>>>
        XSIMD_INLINE batch<T, A> abs(batch<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            typename batch<T, A>::register_type result {};
            for (std::size_t i = 0; i < batch<T, A>::size; ++i)
            {
                result[i] = std::abs(self.data[i]);
            }
            return result;
        }

        // add
        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE batch<T, A> add(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            return self.data + other.data;
        }

        // all
        template <class A, class T>
        XSIMD_INLINE bool all(batch_bool<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            for (std::size_t i = 0; i < batch_bool<T, A>::size; ++i)
            {
                if (self.data[i] == 0)
                {
                    return false;
                }
            }
            return true;
        }

        // any
        template <class A, class T>
        XSIMD_INLINE bool any(batch_bool<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            for (std::size_t i = 0; i < batch_bool<T, A>::size; ++i)
            {
                if (self.data[i] != 0)
                {
                    return true;
                }
            }
            return false;
        }

        // batch_bool_cast
        template <class A, class T_out, class T_in>
        XSIMD_INLINE batch_bool<T_out, A> batch_bool_cast(batch_bool<T_in, A> const& self, batch_bool<T_out, A> const&, requires_arch<loongarch>) noexcept
        {
            using result_type = typename batch_bool<T_out, A>::register_type;
            return detail::loongarch_bit_cast<result_type>(self.data);
        }

        // bitwise operations
        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_and(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            auto bits = detail::loongarch_to_bits(self) & detail::loongarch_to_bits(other);
            return detail::loongarch_from_bits<T, A>(bits);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_and(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            return self.data & other.data;
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_andnot(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            auto bits = detail::loongarch_to_bits(self) & ~detail::loongarch_to_bits(other);
            return detail::loongarch_from_bits<T, A>(bits);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_andnot(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            return self.data & ~other.data;
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_not(batch<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            auto bits = ~detail::loongarch_to_bits(self);
            return detail::loongarch_from_bits<T, A>(bits);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_not(batch_bool<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            return ~self.data;
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_or(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            auto bits = detail::loongarch_to_bits(self) | detail::loongarch_to_bits(other);
            return detail::loongarch_from_bits<T, A>(bits);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_or(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            return self.data | other.data;
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_xor(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            auto bits = detail::loongarch_to_bits(self) ^ detail::loongarch_to_bits(other);
            return detail::loongarch_from_bits<T, A>(bits);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> bitwise_xor(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            return self.data ^ other.data;
        }

        template <class A, class T_in, class T_out>
        XSIMD_INLINE batch<T_out, A> bitwise_cast(batch<T_in, A> const& self, batch<T_out, A> const&, requires_arch<loongarch>) noexcept
        {
            using result_type = typename batch<T_out, A>::register_type;
            return detail::loongarch_bit_cast<result_type>(self.data);
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> bitwise_lshift(batch<T, A> const& self, std::int32_t other, requires_arch<loongarch>) noexcept
        {
            return self.data << other;
        }

        template <class A, class T, class = std::enable_if_t<std::is_integral_v<T>>>
        XSIMD_INLINE batch<T, A> bitwise_rshift(batch<T, A> const& self, std::int32_t other, requires_arch<loongarch>) noexcept
        {
            return self.data >> other;
        }

        // div
        template <class A, class T, class = std::enable_if_t<std::is_floating_point_v<T>>>
        XSIMD_INLINE batch<T, A> div(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            return self.data / other.data;
        }

        // broadcast
        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE batch<T, A> broadcast(T value, requires_arch<loongarch>) noexcept
        {
            typename batch<T, A>::register_type result {};
            for (std::size_t i = 0; i < batch<T, A>::size; ++i)
            {
                result[i] = value;
            }
            return result;
        }

        // comparisons
        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            using result_type = typename batch_bool<T, A>::register_type;
            return detail::loongarch_bit_cast<result_type>(self.data == other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> eq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            return self.data == other.data;
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> lt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            using result_type = typename batch_bool<T, A>::register_type;
            return detail::loongarch_bit_cast<result_type>(self.data < other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> le(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            using result_type = typename batch_bool<T, A>::register_type;
            return detail::loongarch_bit_cast<result_type>(self.data <= other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            using result_type = typename batch_bool<T, A>::register_type;
            return detail::loongarch_bit_cast<result_type>(self.data > other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> ge(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            using result_type = typename batch_bool<T, A>::register_type;
            return detail::loongarch_bit_cast<result_type>(self.data >= other.data);
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> neq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            return bitwise_not(eq(self, other, loongarch {}), loongarch {});
        }

        template <class A, class T>
        XSIMD_INLINE batch_bool<T, A> neq(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            return self.data != other.data;
        }

        // first
        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE T first(batch<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            return self.data[0];
        }

        template <class A, class T>
        XSIMD_INLINE std::complex<T> first(batch<std::complex<T>, A> const& self, requires_arch<loongarch>) noexcept
        {
            return { self.real().data[0], self.imag().data[0] };
        }

        // horizontal add of rows
        template <class A, class T>
        XSIMD_INLINE batch<T, A> haddp(batch<T, A> const* row, requires_arch<loongarch>) noexcept
        {
            typename batch<T, A>::register_type result {};
            for (std::size_t i = 0; i < batch<T, A>::size; ++i)
            {
                T value = T(0);
                for (std::size_t j = 0; j < batch<T, A>::size; ++j)
                {
                    value += row[i].data[j];
                }
                result[i] = value;
            }
            return result;
        }

        // load
        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE batch<T, A> load_unaligned(T const* mem, convert<T>, requires_arch<loongarch>) noexcept
        {
            typename batch<T, A>::register_type result;
            __builtin_memcpy(&result, mem, sizeof(result));
            return result;
        }

        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE batch<T, A> load_aligned(T const* mem, convert<T>, requires_arch<loongarch>) noexcept
        {
            return load_unaligned<A>(mem, convert<T> {}, loongarch {});
        }

        // load/store complex helpers
        namespace detail
        {
            template <class A, class T>
            XSIMD_INLINE batch<std::complex<T>, A> load_complex(batch<T, A> const& first_chunk, batch<T, A> const& second_chunk, requires_arch<loongarch>) noexcept
            {
                constexpr std::size_t size = batch<T, A>::size;
                std::array<T, size> real {};
                std::array<T, size> imag {};
                for (std::size_t i = 0; i < size; ++i)
                {
                    const std::size_t real_index = 2 * i;
                    const std::size_t imag_index = real_index + 1;
                    real[i] = real_index < size ? first_chunk.data[real_index] : second_chunk.data[real_index - size];
                    imag[i] = imag_index < size ? first_chunk.data[imag_index] : second_chunk.data[imag_index - size];
                }
                return { load_unaligned<A>(real.data(), convert<T> {}, loongarch {}),
                         load_unaligned<A>(imag.data(), convert<T> {}, loongarch {}) };
            }

            template <class A, class T>
            XSIMD_INLINE batch<T, A> complex_low(batch<std::complex<T>, A> const& self, requires_arch<loongarch>) noexcept
            {
                constexpr std::size_t size = batch<T, A>::size;
                std::array<T, size> result {};
                for (std::size_t i = 0; i < size; ++i)
                {
                    const std::size_t source = i / 2;
                    result[i] = (i % 2 == 0) ? self.real().data[source] : self.imag().data[source];
                }
                return load_unaligned<A>(result.data(), convert<T> {}, loongarch {});
            }

            template <class A, class T>
            XSIMD_INLINE batch<T, A> complex_high(batch<std::complex<T>, A> const& self, requires_arch<loongarch>) noexcept
            {
                constexpr std::size_t size = batch<T, A>::size;
                std::array<T, size> result {};
                for (std::size_t i = 0; i < size; ++i)
                {
                    const std::size_t interleaved_index = size + i;
                    const std::size_t source = interleaved_index / 2;
                    result[i] = (interleaved_index % 2 == 0) ? self.real().data[source] : self.imag().data[source];
                }
                return load_unaligned<A>(result.data(), convert<T> {}, loongarch {});
            }
        }

        // max/min
        template <class A, class T>
        XSIMD_INLINE batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            typename batch<T, A>::register_type result {};
            for (std::size_t i = 0; i < batch<T, A>::size; ++i)
            {
                result[i] = self.data[i] < other.data[i] ? other.data[i] : self.data[i];
            }
            return result;
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            typename batch<T, A>::register_type result {};
            for (std::size_t i = 0; i < batch<T, A>::size; ++i)
            {
                result[i] = self.data[i] < other.data[i] ? self.data[i] : other.data[i];
            }
            return result;
        }

        // mul/neg
        template <class A, class T>
        XSIMD_INLINE batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            return self.data * other.data;
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> neg(batch<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            return -self.data;
        }

        // rsqrt
        template <class A, class T, class = std::enable_if_t<std::is_floating_point_v<T>>>
        XSIMD_INLINE batch<T, A> rsqrt(batch<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            typename batch<T, A>::register_type result {};
            for (std::size_t i = 0; i < batch<T, A>::size; ++i)
            {
                result[i] = T(1) / std::sqrt(self.data[i]);
            }
            return result;
        }

        // isnan
        template <class A, class T, class = std::enable_if_t<std::is_floating_point_v<T>>>
        XSIMD_INLINE batch_bool<T, A> isnan(batch<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            return neq(self, self, loongarch {});
        }

        // select
        template <class A, class T>
        XSIMD_INLINE batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<loongarch>) noexcept
        {
            using bits_type = detail::loongarch_unsigned_register_t<T, A>;
            auto mask = detail::loongarch_bit_cast<bits_type>(cond.data);
            auto true_bits = detail::loongarch_to_bits(true_br);
            auto false_bits = detail::loongarch_to_bits(false_br);
            return detail::loongarch_from_bits<T, A>((mask & true_bits) | (~mask & false_bits));
        }

        template <class A, class T, bool... Values>
        XSIMD_INLINE batch<T, A> select(batch_bool_constant<T, A, Values...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<loongarch>) noexcept
        {
            return select(batch_bool<T, A> { Values... }, true_br, false_br, loongarch {});
        }

        // set
        template <class A, class T, class... Values>
        XSIMD_INLINE batch<T, A> set(batch<T, A> const&, requires_arch<loongarch>, Values... values) noexcept
        {
            static_assert(sizeof...(Values) == batch<T, A>::size, "consistent init");
            return typename batch<T, A>::register_type { static_cast<T>(values)... };
        }

        template <class A, class T, class... Values>
        XSIMD_INLINE batch<std::complex<T>, A> set(batch<std::complex<T>, A> const&, requires_arch<loongarch>, Values... values) noexcept
        {
            return batch<std::complex<T>, A>(set(batch<T, A> {}, loongarch {}, values.real()...),
                                              set(batch<T, A> {}, loongarch {}, values.imag()...));
        }

        template <class A, class T, class... Values>
        XSIMD_INLINE batch_bool<T, A> set(batch_bool<T, A> const&, requires_arch<loongarch>, Values... values) noexcept
        {
            static_assert(sizeof...(Values) == batch_bool<T, A>::size, "consistent init");
            using value_type = sized_uint_t<sizeof(T)>;
            return typename batch_bool<T, A>::register_type { static_cast<value_type>(values ? ~value_type(0) : value_type(0))... };
        }

        // sqrt
        template <class A, class T, class = std::enable_if_t<std::is_floating_point_v<T>>>
        XSIMD_INLINE batch<T, A> sqrt(batch<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            typename batch<T, A>::register_type result {};
            for (std::size_t i = 0; i < batch<T, A>::size; ++i)
            {
                result[i] = std::sqrt(self.data[i]);
            }
            return result;
        }

        // byte slides
        template <std::size_t N, class A, class T>
        XSIMD_INLINE batch<T, A> slide_left(batch<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            static_assert(N <= A::alignment(), "invalid byte slide");
            std::array<unsigned char, A::alignment()> input {};
            std::array<unsigned char, A::alignment()> output {};
            __builtin_memcpy(input.data(), &self.data, input.size());
            for (std::size_t i = N; i < output.size(); ++i)
            {
                output[i] = input[i - N];
            }
            typename batch<T, A>::register_type result;
            __builtin_memcpy(&result, output.data(), output.size());
            return result;
        }

        template <std::size_t N, class A, class T>
        XSIMD_INLINE batch<T, A> slide_right(batch<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            static_assert(N <= A::alignment(), "invalid byte slide");
            std::array<unsigned char, A::alignment()> input {};
            std::array<unsigned char, A::alignment()> output {};
            __builtin_memcpy(input.data(), &self.data, input.size());
            for (std::size_t i = 0; i + N < output.size(); ++i)
            {
                output[i] = input[i + N];
            }
            typename batch<T, A>::register_type result;
            __builtin_memcpy(&result, output.data(), output.size());
            return result;
        }

        // store
        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE void store_unaligned(T* mem, batch<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            __builtin_memcpy(mem, &self.data, sizeof(self.data));
        }

        template <class A, class T, class = std::enable_if_t<std::is_scalar_v<T>>>
        XSIMD_INLINE void store_aligned(T* mem, batch<T, A> const& self, requires_arch<loongarch>) noexcept
        {
            store_unaligned<A>(mem, self, loongarch {});
        }

        // sub
        template <class A, class T>
        XSIMD_INLINE batch<T, A> sub(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            return self.data - other.data;
        }

        // zip
        template <class A, class T>
        XSIMD_INLINE batch<T, A> zip_lo(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            typename batch<T, A>::register_type result {};
            constexpr std::size_t half = batch<T, A>::size / 2;
            for (std::size_t i = 0; i < half; ++i)
            {
                result[2 * i] = self.data[i];
                result[2 * i + 1] = other.data[i];
            }
            return result;
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> zip_hi(batch<T, A> const& self, batch<T, A> const& other, requires_arch<loongarch>) noexcept
        {
            typename batch<T, A>::register_type result {};
            constexpr std::size_t half = batch<T, A>::size / 2;
            for (std::size_t i = 0; i < half; ++i)
            {
                result[2 * i] = self.data[i + half];
                result[2 * i + 1] = other.data[i + half];
            }
            return result;
        }
    }
}

#endif
