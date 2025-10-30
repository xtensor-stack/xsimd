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

#ifndef XSIMD_BATCH_CONSTANT_HPP
#define XSIMD_BATCH_CONSTANT_HPP

#include <utility>

#include "./xsimd_batch.hpp"
#include "./xsimd_utils.hpp"

namespace xsimd
{
    /**
     * @brief batch of boolean constant
     *
     * Abstract representation of a batch of boolean constants.
     *
     * @tparam batch_type the type of the associated batch values.
     * @tparam Values boolean constant represented by this batch
     **/
    template <typename T, class A, bool... Values>
    struct batch_bool_constant
    {
        using batch_type = batch_bool<T, A>;
        static constexpr std::size_t size = sizeof...(Values);
        using value_type = bool;
        static_assert(sizeof...(Values) == batch_type::size, "consistent batch size");

    public:
        /**
         * @brief Generate a batch of @p batch_type from this @p batch_bool_constant
         */
        constexpr batch_type as_batch_bool() const noexcept { return { Values... }; }

        /**
         * @brief Generate a batch of @p integers from this @p batch_bool_constant
         */
        constexpr batch<as_integer_t<T>, A> as_batch() const noexcept { return { -as_integer_t<T>(Values)... }; } // the minus is important!
        /**
         * @brief Generate a batch of @p batch_type from this @p batch_bool_constant
         */
        constexpr operator batch_type() const noexcept { return as_batch_bool(); }

        constexpr bool get(size_t i) const noexcept
        {
            return std::array<value_type, size> { { Values... } }[i];
        }

        static constexpr int mask() noexcept
        {
            return mask_helper(0, static_cast<int>(Values)...);
        }

        static constexpr bool none() noexcept
        {
            return truncated_mask() == 0u;
        }

        static constexpr bool any() noexcept
        {
            return !none();
        }

        static constexpr bool all() noexcept
        {
            return truncated_mask() == low_mask(size);
        }

        static constexpr std::size_t countr_zero() noexcept
        {
            return countr_zero_impl(truncated_mask(), size);
        }

        static constexpr std::size_t countl_zero() noexcept
        {
            return countl_zero_impl(truncated_mask(), size);
        }

        static constexpr std::size_t countr_one() noexcept
        {
            return countr_one_impl(truncated_mask(), size);
        }

        static constexpr std::size_t countl_one() noexcept
        {
            return countl_one_impl(truncated_mask(), size);
        }

    private:
        static constexpr int mask_helper(int acc) noexcept { return acc; }

        template <class... Tys>
        static constexpr int mask_helper(int acc, int mask, Tys... masks) noexcept
        {
            return mask_helper(acc | mask, (masks << 1)...);
        }

        struct logical_or
        {
            constexpr bool operator()(bool x, bool y) const { return x || y; }
        };
        struct logical_and
        {
            constexpr bool operator()(bool x, bool y) const { return x && y; }
        };
        struct logical_xor
        {
            constexpr bool operator()(bool x, bool y) const { return x ^ y; }
        };

        template <class F, class SelfPack, class OtherPack, size_t... Indices>
        static constexpr batch_bool_constant<T, A, F()(std::tuple_element<Indices, SelfPack>::type::value, std::tuple_element<Indices, OtherPack>::type::value)...>
        apply(detail::index_sequence<Indices...>)
        {
            return {};
        }

        template <class F, bool... OtherValues>
        static constexpr auto apply(batch_bool_constant<T, A, Values...>, batch_bool_constant<T, A, OtherValues...>)
            -> decltype(apply<F, std::tuple<std::integral_constant<bool, Values>...>, std::tuple<std::integral_constant<bool, OtherValues>...>>(detail::make_index_sequence<sizeof...(Values)>()))
        {
            static_assert(sizeof...(Values) == sizeof...(OtherValues), "compatible constant batches");
            return apply<F, std::tuple<std::integral_constant<bool, Values>...>, std::tuple<std::integral_constant<bool, OtherValues>...>>(detail::make_index_sequence<sizeof...(Values)>());
        }

    public:
#define MAKE_BINARY_OP(OP, NAME)                                                      \
    template <bool... OtherValues>                                                    \
    constexpr auto operator OP(batch_bool_constant<T, A, OtherValues...> other) const \
        -> decltype(apply<NAME>(*this, other))                                        \
    {                                                                                 \
        return apply<NAME>(*this, other);                                             \
    }

        MAKE_BINARY_OP(|, logical_or)
        MAKE_BINARY_OP(||, logical_or)
        MAKE_BINARY_OP(&, logical_and)
        MAKE_BINARY_OP(&&, logical_and)
        MAKE_BINARY_OP(^, logical_xor)

#undef MAKE_BINARY_OP

        constexpr batch_bool_constant<T, A, !Values...> operator!() const
        {
            return {};
        }

        constexpr batch_bool_constant<T, A, !Values...> operator~() const
        {
            return {};
        }

    private:
        // Build a 64-bit mask from Values... (LSB = index 0)
        template <std::size_t I, bool... Remaining>
        struct build_bits_helper;

        template <std::size_t I>
        struct build_bits_helper<I>
        {
            static constexpr uint64_t value = 0u;
        };

        template <std::size_t I, bool Current, bool... Remaining>
        struct build_bits_helper<I, Current, Remaining...>
        {
            static constexpr uint64_t value = (Current ? (uint64_t(1) << I) : 0u)
                | build_bits_helper<I + 1, Remaining...>::value;
        };

        static constexpr uint64_t bits() noexcept
        {
            return build_bits_helper<0, Values...>::value;
        }
        static constexpr uint64_t low_mask(std::size_t k) noexcept
        {
            return (k >= 64u) ? ~uint64_t(0) : ((uint64_t(1) << k) - 1u);
        }
        static constexpr uint64_t truncated_mask() noexcept
        {
            return bits() & low_mask(size);
        }
        static constexpr std::size_t countr_zero_impl(uint64_t v, std::size_t n) noexcept
        {
            return (n == 0 || (v & 1u) != 0u) ? 0u : (1u + countr_zero_impl(v >> 1, n - 1));
        }
        static constexpr std::size_t countr_one_impl(uint64_t v, std::size_t n) noexcept
        {
            return (n == 0 || (v & 1u) == 0u) ? 0u : (1u + countr_one_impl(v >> 1, n - 1));
        }
        static constexpr std::size_t countl_zero_impl(uint64_t v, std::size_t n) noexcept
        {
            return (n == 0) ? 0u : ((((v >> (n - 1)) & 1u) != 0u) ? 0u : (1u + countl_zero_impl(v, n - 1)));
        }
        static constexpr std::size_t countl_one_impl(uint64_t v, std::size_t n) noexcept
        {
            return (n == 0) ? 0u : ((((v >> (n - 1)) & 1u) == 0u) ? 0u : (1u + countl_one_impl(v, n - 1)));
        }
    };

    namespace detail
    {
        template <class A2, std::size_t BeginIdx, typename T, class A, bool... Values, std::size_t... Is>
        XSIMD_INLINE constexpr auto splice_impl(index_sequence<Is...>) noexcept
            -> batch_bool_constant<
                T, A2,
                std::tuple_element<BeginIdx + Is,
                                   std::tuple<std::integral_constant<bool, Values>...>>::type::value...>
        {
            return {};
        }

        template <class A2, std::size_t Begin, std::size_t End,
                  typename T, class A, bool... Values,
                  std::size_t N = (End >= Begin ? (End - Begin) : 0)>
        XSIMD_INLINE constexpr auto splice(batch_bool_constant<T, A, Values...>) noexcept
            -> decltype(splice_impl<A2, Begin, T, A, Values...>(make_index_sequence<N>()))
        {
            static_assert(Begin <= End, "splice: Begin must be <= End");
            static_assert(End <= sizeof...(Values), "splice: End must be <= size");
            static_assert(N == batch_bool<T, A2>::size, "splice: target arch size must match submask length");
            return splice_impl<A2, Begin, T, A, Values...>(make_index_sequence<N>());
        }

        template <class A2, typename T, class A, bool... Values>
        XSIMD_INLINE constexpr auto lower_half(batch_bool_constant<T, A, Values...>) noexcept
            -> decltype(splice_impl<A2, 0, T, A, Values...>(make_index_sequence<sizeof...(Values) / 2>()))
        {
            static_assert(sizeof...(Values) % 2 == 0, "lower_half requires even size");
            static_assert(batch_bool<T, A2>::size == sizeof...(Values) / 2,
                          "lower_half: target arch size must match submask length");
            return splice_impl<A2, 0, T, A, Values...>(make_index_sequence<sizeof...(Values) / 2>());
        }

        template <class A2, typename T, class A, bool... Values>
        XSIMD_INLINE constexpr auto upper_half(batch_bool_constant<T, A, Values...>) noexcept
            -> decltype(splice_impl<A2, sizeof...(Values) / 2, T, A, Values...>(make_index_sequence<sizeof...(Values) / 2>()))
        {
            static_assert(sizeof...(Values) % 2 == 0, "upper_half requires even size");
            static_assert(batch_bool<T, A2>::size == sizeof...(Values) / 2,
                          "upper_half: target arch size must match submask length");
            return splice_impl<A2, sizeof...(Values) / 2, T, A, Values...>(make_index_sequence<sizeof...(Values) / 2>());
        }
    } // namespace detail

    /**
     * @brief batch of integral constants
     *
     * Abstract representation of a batch of integral constants.
     *
     * @tparam batch_type the type of the associated batch values.
     * @tparam Values constants represented by this batch
     **/
    template <typename T, class A, T... Values>
    struct batch_constant
    {
        static constexpr std::size_t size = sizeof...(Values);
        using batch_type = batch<T, A>;
        using value_type = typename batch_type::value_type;
        static_assert(sizeof...(Values) == batch_type::size, "consistent batch size");

        /**
         * @brief Generate a batch of @p batch_type from this @p batch_constant
         */
        XSIMD_INLINE batch_type as_batch() const noexcept { return { Values... }; }

        /**
         * @brief Generate a batch of @p batch_type from this @p batch_constant
         */
        XSIMD_INLINE operator batch_type() const noexcept { return as_batch(); }

        /**
         * @brief Get the @p i th element of this @p batch_constant
         */
        constexpr T get(size_t i) const noexcept
        {
            return get(i, std::array<T, size> { Values... });
        }

    private:
        constexpr T get(size_t i, std::array<T, size> const& values) const noexcept
        {
            return values[i];
        }

        struct arithmetic_add
        {
            constexpr T operator()(T x, T y) const { return x + y; }
        };
        struct arithmetic_sub
        {
            constexpr T operator()(T x, T y) const { return x - y; }
        };
        struct arithmetic_mul
        {
            constexpr T operator()(T x, T y) const { return x * y; }
        };
        struct arithmetic_div
        {
            constexpr T operator()(T x, T y) const { return x / y; }
        };
        struct arithmetic_mod
        {
            constexpr T operator()(T x, T y) const { return x % y; }
        };
        struct binary_and
        {
            constexpr T operator()(T x, T y) const { return x & y; }
        };
        struct binary_or
        {
            constexpr T operator()(T x, T y) const { return x | y; }
        };
        struct binary_xor
        {
            constexpr T operator()(T x, T y) const { return x ^ y; }
        };

        template <class F, class SelfPack, class OtherPack, size_t... Indices>
        static constexpr batch_constant<T, A, F()(std::tuple_element<Indices, SelfPack>::type::value, std::tuple_element<Indices, OtherPack>::type::value)...>
        apply(detail::index_sequence<Indices...>)
        {
            return {};
        }

        template <class F, T... OtherValues>
        static constexpr auto apply(batch_constant<T, A, Values...>, batch_constant<T, A, OtherValues...>)
            -> decltype(apply<F, std::tuple<std::integral_constant<T, Values>...>, std::tuple<std::integral_constant<T, OtherValues>...>>(detail::make_index_sequence<sizeof...(Values)>()))
        {
            static_assert(sizeof...(Values) == sizeof...(OtherValues), "compatible constant batches");
            return apply<F, std::tuple<std::integral_constant<T, Values>...>, std::tuple<std::integral_constant<T, OtherValues>...>>(detail::make_index_sequence<sizeof...(Values)>());
        }

    public:
#define MAKE_BINARY_OP(OP, NAME)                                                 \
    template <T... OtherValues>                                                  \
    constexpr auto operator OP(batch_constant<T, A, OtherValues...> other) const \
        -> decltype(apply<NAME>(*this, other))                                   \
    {                                                                            \
        return apply<NAME>(*this, other);                                        \
    }

        MAKE_BINARY_OP(+, arithmetic_add)
        MAKE_BINARY_OP(-, arithmetic_sub)
        MAKE_BINARY_OP(*, arithmetic_mul)
        MAKE_BINARY_OP(/, arithmetic_div)
        MAKE_BINARY_OP(%, arithmetic_mod)
        MAKE_BINARY_OP(&, binary_and)
        MAKE_BINARY_OP(|, binary_or)
        MAKE_BINARY_OP(^, binary_xor)

#undef MAKE_BINARY_OP

        constexpr batch_constant<T, A, (T)-Values...> operator-() const
        {
            return {};
        }

        constexpr batch_constant<T, A, (T) + Values...> operator+() const
        {
            return {};
        }

        constexpr batch_constant<T, A, (T)~Values...> operator~() const
        {
            return {};
        }
    };

    namespace detail
    {
        template <typename T, class G, class A, std::size_t... Is>
        XSIMD_INLINE constexpr auto make_batch_constant(detail::index_sequence<Is...>) noexcept
            -> batch_constant<T, A, (T)G::get(Is, sizeof...(Is))...>
        {
            return {};
        }
        template <typename T, class G, class A, std::size_t... Is>
        XSIMD_INLINE constexpr auto make_batch_bool_constant(detail::index_sequence<Is...>) noexcept
            -> batch_bool_constant<T, A, G::get(Is, sizeof...(Is))...>
        {
            return {};
        }

    } // namespace detail

    /**
     * @brief Build a @c batch_constant out of a generator function
     *
     * @tparam batch_type type of the (non-constant) batch to build
     * @tparam G type used to generate that batch. That type must have a static
     * member @c get that's used to generate the batch constant. Conversely, the
     * generated batch_constant has value `{G::get(0, batch_size), ... , G::get(batch_size - 1, batch_size)}`
     *
     * The following generator produces a batch of `(n - 1, 0, 1, ... n-2)`
     *
     * @code
     * struct Rot
     * {
     *     static constexpr unsigned get(unsigned i, unsigned n)
     *     {
     *         return (i + n - 1) % n;
     *     }
     * };
     * @endcode
     */
    template <typename T, class G, class A = default_arch>
    XSIMD_INLINE constexpr auto make_batch_constant() noexcept -> decltype(detail::make_batch_constant<T, G, A>(detail::make_index_sequence<batch<T, A>::size>()))
    {
        return detail::make_batch_constant<T, G, A>(detail::make_index_sequence<batch<T, A>::size>());
    }

    template <typename T, class G, class A = default_arch>
    XSIMD_INLINE constexpr auto make_batch_bool_constant() noexcept
        -> decltype(detail::make_batch_bool_constant<T, G, A>(
            detail::make_index_sequence<batch<T, A>::size>()))
    {
        return detail::make_batch_bool_constant<T, G, A>(
            detail::make_index_sequence<batch<T, A>::size>());
    }

} // namespace xsimd

#endif
