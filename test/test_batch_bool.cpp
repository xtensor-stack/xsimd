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

#include "xsimd/xsimd.hpp"
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include <functional>
#include <vector>

#include "test_utils.hpp"

namespace xsimd
{

    int popcount(int v)
    {
        // from https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
        int c; // c accumulates the total bits set in v
        for (c = 0; v; c++)
        {
            v &= v - 1; // clear the least significant bit set
        }
        return c;
    }

    template <class T, std::size_t N>
    struct get_bool_base
    {
        using vector_type = std::array<bool, N>;

        std::vector<vector_type> almost_all_false()
        {
            std::vector<vector_type> vectors;
            vectors.reserve(N);
            for (size_t i = 0; i < N; ++i)
            {
                vector_type v;
                v.fill(false);
                v[i] = true;
                vectors.push_back(std::move(v));
            }
            return vectors;
        }

        std::vector<vector_type> almost_all_true()
        {
            auto vectors = almost_all_false();
            flip(vectors);
            return vectors;
        }

        void flip(vector_type& vec)
        {
            std::transform(vec.begin(), vec.end(), vec.begin(), std::logical_not<bool> {});
        }

        void flip(std::vector<vector_type>& vectors)
        {
            for (auto& vec : vectors)
            {
                flip(vec);
            }
        }
    };

    template <class T, size_t N = T::size>
    struct get_bool;

    template <class T>
    struct get_bool<batch_bool<T>, 1> : public get_bool_base<T, 1>
    {
        using type = batch_bool<T>;
        type all_true = type(true);
        type all_false = type(false);
        type half = { 0 };
        type ihalf = { 1 };
        type interspersed = { 0 };
    };

    template <class T>
    struct get_bool<batch_bool<T>, 2> : public get_bool_base<T, 2>
    {
        using type = batch_bool<T>;
        type all_true = type(true);
        type all_false = type(false);
        type half = { 0, 1 };
        type ihalf = { 1, 0 };
        type interspersed = { 0, 1 };
    };

    template <class T>
    struct get_bool<batch_bool<T>, 4> : public get_bool_base<T, 4>
    {
        using type = batch_bool<T>;

        type all_true = true;
        type all_false = false;
        type half = { 0, 0, 1, 1 };
        type ihalf = { 1, 1, 0, 0 };
        type interspersed = { 0, 1, 0, 1 };
    };

    template <class T>
    struct get_bool<batch_bool<T>, 8> : public get_bool_base<T, 8>
    {
        using type = batch_bool<T>;
        type all_true = true;
        type all_false = false;
        type half = { 0, 0, 0, 0, 1, 1, 1, 1 };
        type ihalf = { 1, 1, 1, 1, 0, 0, 0, 0 };
        type interspersed = { 0, 1, 0, 1, 0, 1, 0, 1 };
    };

    template <class T>
    struct get_bool<batch_bool<T>, 16> : public get_bool_base<T, 16>
    {
        using type = batch_bool<T>;
        type all_true = true;
        type all_false = false;
        type half = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 };
        type ihalf = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
        type interspersed = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };
    };

    template <class T>
    struct get_bool<batch_bool<T>, 32> : public get_bool_base<T, 32>
    {
        using type = batch_bool<T>;
        type all_true = true;
        type all_false = false;
        type half = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        type ihalf = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        type interspersed = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };
    };

    template <class T>
    struct get_bool<batch_bool<T>, 64> : public get_bool_base<T, 64>
    {
        using type = batch_bool<T>;
        type all_true = true;
        type all_false = false;
        type half = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        type ihalf = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        type interspersed = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };
    };

}

template <class T>
struct batch_bool_test
{
    using batch_type = T;
    using value_type = typename T::value_type;
    static constexpr size_t size = T::size;
    using batch_bool_type = typename T::batch_bool_type;
    using array_type = std::array<value_type, size>;
    using bool_array_type = std::array<bool, size>;

    // Compile-time check helpers for batch_bool_constant masks
    template <class B, class Enable = void>
    struct xsimd_ct_mask_checker;

    // Small masks: safe to compare numeric masks at compile time
    template <class B>
    struct xsimd_ct_mask_checker<B, typename std::enable_if<(B::size <= 31)>::type>
    {
        static constexpr std::size_t sum_indices(uint64_t bits, std::size_t index, std::size_t remaining)
        {
            return remaining == 0
                ? 0u
                : ((bits & 1u ? index : 0u) + sum_indices(bits >> 1, index + 1, remaining - 1));
        }

        struct ct_false
        {
            static constexpr bool get(std::size_t, std::size_t) { return false; }
        };
        struct ct_true
        {
            static constexpr bool get(std::size_t, std::size_t) { return true; }
        };
        struct ct_prefix1
        {
            static constexpr bool get(std::size_t i, std::size_t) { return i < 1; }
        };
        struct ct_suffix1
        {
            static constexpr bool get(std::size_t i, std::size_t n) { return i >= (n - 1); }
        };
        struct ct_ends
        {
            static constexpr bool get(std::size_t i, std::size_t n) { return (i < 1) || (i >= (n - 1)); }
        };

        static void run()
        {
            using value_type = typename B::value_type;
            using arch_type = typename B::arch_type;
            constexpr auto m_zero = xsimd::make_batch_bool_constant<value_type, ct_false, arch_type>();
            constexpr auto m_one = xsimd::make_batch_bool_constant<value_type, ct_true, arch_type>();
            constexpr auto m_prefix = xsimd::make_batch_bool_constant<value_type, ct_prefix1, arch_type>();
            constexpr auto m_suffix = xsimd::make_batch_bool_constant<value_type, ct_suffix1, arch_type>();
            constexpr auto m_ends = xsimd::make_batch_bool_constant<value_type, ct_ends, arch_type>();

            static_assert((m_zero | m_one).mask() == m_one.mask(), "0|1 == 1");
            static_assert((m_zero & m_one).mask() == m_zero.mask(), "0&1 == 0");
            static_assert((m_zero ^ m_zero).mask() == m_zero.mask(), "0^0 == 0");
            static_assert((m_one ^ m_one).mask() == m_zero.mask(), "1^1 == 0");

            static_assert((m_prefix | m_suffix).mask() == m_ends.mask(), "prefix|suffix == ends");
            static_assert(B::size == 1 || (m_prefix & m_suffix).mask() == m_zero.mask(), "prefix&suffix == 0 when size>1");

            static_assert(m_zero.none(), "zero mask none");
            static_assert(!m_zero.any(), "zero mask any");
            static_assert(!m_zero.all(), "zero mask all");
            static_assert(m_zero.countr_zero() == B::size, "zero mask trailing zeros");
            static_assert(m_zero.countl_zero() == B::size, "zero mask leading zeros");

            static_assert(m_one.all(), "all mask all");
            static_assert(m_one.any(), "all mask any");
            static_assert(!m_one.none(), "all mask none");
            static_assert(m_one.countr_zero() == 0, "all mask trailing zeros");
            static_assert(m_one.countl_zero() == 0, "all mask leading zeros");

            constexpr std::size_t prefix_zero = m_prefix.countr_zero();
            constexpr std::size_t prefix_one = m_prefix.countr_one();
            static_assert(prefix_zero == 0, "prefix mask zero leading zeros from LSB");
            static_assert(prefix_one == 1 || B::size == 0, "prefix mask trailing ones count");

            constexpr std::size_t suffix_zero = m_suffix.countl_zero();
            constexpr std::size_t suffix_one = m_suffix.countl_one();
            static_assert(suffix_zero == 0 || B::size == 0, "suffix mask leading zeros count");
            static_assert(suffix_one == 1 || B::size == 0, "suffix mask trailing ones count");

            constexpr std::size_t ends_bits = m_ends.truncated_mask();
            constexpr std::size_t ends_sum = sum_indices(ends_bits, 0u, B::size);
            static_assert((B::size <= 1 && ends_sum == 0) || (B::size > 1 && ends_sum == (B::size - 1)), "ends index sum coverage");
        }
    };

    // Large masks: avoid calling mask() in constant expressions
    template <class B>
    struct xsimd_ct_mask_checker<B, typename std::enable_if<(B::size > 31)>::type>
    {
        static void run() { }
    };

    array_type lhs;
    array_type rhs;
    bool_array_type all_true;
    bool_array_type ba;

    batch_bool_test()
    {
        for (size_t i = 0; i < size; ++i)
        {
            lhs[i] = value_type(i);
            rhs[i] = i == 0 % 2 ? lhs[i] : lhs[i] * value_type(2);
            all_true[i] = true;
            ba[i] = i == 0 % 2 ? true : false;
        }
    }

    template <size_t... Is>
    struct pack
    {
    };

    template <typename F, size_t... Values>
    static batch_bool_type make_batch_impl(F&& f, std::integral_constant<size_t, 0>, pack<Values...>)
    {
        return batch_bool_type(bool(f(Values))...);
    }

    template <typename F, size_t I, size_t... Values>
    static batch_bool_type make_batch_impl(F&& f, std::integral_constant<size_t, I>, pack<Values...>)
    {
        return make_batch_impl(std::forward<F>(f), std::integral_constant<size_t, I - 1>(), pack<I - 1, Values...>());
    }

    template <typename F>
    static batch_bool_type make_batch(F&& f)
    {
        return make_batch_impl(std::forward<F>(f), std::integral_constant<size_t, size>(), pack<> {});
    }

    void test_constructors() const
    {
        batch_bool_type a;
        // value uninitialized, cannot test it.
        (void)a;

        {
            bool_array_type res;
            batch_bool_type b(true);
            b.store_unaligned(res.data());
            INFO("batch_bool{value}");
            CHECK_EQ(res, all_true);

            batch_bool_type c { true };
            c.store_unaligned(res.data());
            INFO("batch_bool{value}");
            CHECK_EQ(res, all_true);
        }

        {
            auto f_bool = [](size_t i)
            { return bool(i % 3); };

            bool_array_type res;
            for (size_t i = 0; i < res.size(); i++)
            {
                res[i] = f_bool(i);
            }

            bool_array_type tmp;
            batch_bool_type b0 = make_batch(f_bool);
            b0.store_unaligned(tmp.data());
            INFO("batch_bool(values...)");
            CHECK_EQ(tmp, res);

            batch_bool_type b1 = make_batch(f_bool);
            b1.store_unaligned(tmp.data());
            INFO("batch_bool{values...}");
            CHECK_EQ(tmp, res);
        }
    }

    void test_load_store() const
    {
        bool_array_type res;
        batch_bool_type b(batch_bool_type::load_unaligned(ba.data()));
        b.store_unaligned(res.data());
        CHECK_EQ(res, ba);

        alignas(xsimd::default_arch::alignment()) bool_array_type arhs(this->ba);
        alignas(xsimd::default_arch::alignment()) bool_array_type ares;
        b = batch_bool_type::load_aligned(arhs.data());
        b.store_aligned(ares.data());
        CHECK_EQ(ares, arhs);

        auto bool_g = xsimd::get_bool<batch_bool_type> {};
        // load/store, almost all false
        {
            size_t i = 0;
            for (const auto& vec : bool_g.almost_all_false())
            {
                batch_bool_type b = batch_bool_type::load_unaligned(vec.data());
                batch_bool_type expected = make_batch([i](size_t x)
                                                      { return x == i; });
                i++;
                CHECK_UNARY(xsimd::all(b == expected));
                b.store_unaligned(res.data());
                // Check that the representation is bitwise exact.
                CHECK_UNARY(memcmp(res.data(), vec.data(), sizeof(res)) == 0);
            }
        }

        // load/store, almost all true
        {
            size_t i = 0;
            for (const auto& vec : bool_g.almost_all_true())
            {
                batch_bool_type b = batch_bool_type::load_unaligned(vec.data());
                batch_bool_type expected = make_batch([i](size_t x)
                                                      { return x != i; });
                i++;
                CHECK_UNARY(xsimd::all(b == expected));
                b.store_unaligned(res.data());
                CHECK_UNARY(memcmp(res.data(), vec.data(), sizeof(res)) == 0);
            }
        }
    }

    void test_any_all() const
    {
        auto bool_g = xsimd::get_bool<batch_bool_type> {};
        // any
        {
            auto any_check_false = (batch_lhs() != batch_lhs());
            bool any_res_false = xsimd::any(any_check_false);
            CHECK_FALSE(any_res_false);
            auto any_check_true = (batch_lhs() == batch_rhs());
            bool any_res_true = xsimd::any(any_check_true);
            CHECK_UNARY(any_res_true);

            for (const auto& vec : bool_g.almost_all_false())
            {
                batch_bool_type b = batch_bool_type::load_unaligned(vec.data());
                bool any_res = xsimd::any(b);
                CHECK_UNARY(any_res);
            }

            for (const auto& vec : bool_g.almost_all_true())
            {
                batch_bool_type b = batch_bool_type::load_unaligned(vec.data());
                bool any_res = xsimd::any(b);
                CHECK_UNARY(any_res);
            }
        }
        // all
        {
            auto all_check_false = (batch_lhs() == batch_rhs());
            bool all_res_false = xsimd::all(all_check_false);
            CHECK_FALSE(all_res_false);
            auto all_check_true = (batch_lhs() == batch_lhs());
            bool all_res_true = xsimd::all(all_check_true);
            CHECK_UNARY(all_res_true);

            for (const auto& vec : bool_g.almost_all_false())
            {
                // TODO: implement batch_bool(bool*)
                // It currently compiles (need to understand why) but does not
                // give expected result
                batch_bool_type b = batch_bool_type::load_unaligned(vec.data());
                bool all_res = xsimd::all(b);
                CHECK_FALSE(all_res);
            }

            for (const auto& vec : bool_g.almost_all_true())
            {
                batch_bool_type b = batch_bool_type::load_unaligned(vec.data());
                bool all_res = xsimd::all(b);
                CHECK_FALSE(all_res);
            }
        }
        // none
        {
            auto none_check_false = (batch_lhs() == batch_rhs());
            bool none_res_false = xsimd::none(none_check_false);
            CHECK_FALSE(none_res_false);
            auto none_check_true = (batch_lhs() != batch_lhs());
            bool none_res_true = xsimd::none(none_check_true);
            CHECK_UNARY(none_res_true);

            for (const auto& vec : bool_g.almost_all_false())
            {
                batch_bool_type b = batch_bool_type::load_unaligned(vec.data());
                bool none_res = xsimd::none(b);
                CHECK_FALSE(none_res);
            }

            for (const auto& vec : bool_g.almost_all_true())
            {
                batch_bool_type b = batch_bool_type::load_unaligned(vec.data());
                bool none_res = xsimd::none(b);
                CHECK_FALSE(none_res);
            }
        }
    }

    void test_logical_operations() const
    {
        auto bool_g = xsimd::get_bool<batch_bool_type> {};
        size_t s = size;
        // operator!=
        {
            bool res = xsimd::all(bool_g.half != bool_g.ihalf);
            CHECK_UNARY(res);
        }
        // operator==
        {
            CHECK_BATCH_EQ(bool_g.half, !bool_g.ihalf);
        }
        // operator &&
        {
            batch_bool_type res = bool_g.half && bool_g.ihalf;
            bool_array_type ares;
            res.store_unaligned(ares.data());
            size_t nb_false = std::count(ares.cbegin(), ares.cend(), false);
            CHECK_EQ(nb_false, s);
        }
        // operator ||
        {
            batch_bool_type res = bool_g.half || bool_g.ihalf;
            bool_array_type ares;
            res.store_unaligned(ares.data());
            size_t nb_true = std::count(ares.cbegin(), ares.cend(), true);
            CHECK_EQ(nb_true, s);
        }
        // operator ^
        {
            batch_bool_type res = bool_g.half ^ bool_g.ihalf;
            bool_array_type ares;
            res.store_unaligned(ares.data());
            size_t nb_true = std::count(ares.cbegin(), ares.cend(), true);
            CHECK_EQ(nb_true, s);
        }
        // bitwise_andnot
        {
            batch_bool_type res = xsimd::bitwise_andnot(bool_g.half, bool_g.half);
            bool_array_type ares;
            res.store_unaligned(ares.data());
            size_t nb_false = std::count(ares.cbegin(), ares.cend(), false);
            CHECK_EQ(nb_false, s);
        }
    }

    void test_bitwise_operations() const
    {
        auto bool_g = xsimd::get_bool<batch_bool_type> {};
        // operator version
        {
            INFO("operator~");
            CHECK_BATCH_EQ(bool_g.half, ~bool_g.ihalf);
        }
        {
            INFO("operator|");
            CHECK_BATCH_EQ(bool_g.half | bool_g.ihalf, bool_g.all_true);
        }
        {
            INFO("operator&");
            CHECK_BATCH_EQ(bool_g.half & bool_g.ihalf, bool_g.all_false);
        }
        {
            INFO("operator^");
            CHECK_BATCH_EQ(bool_g.half ^ bool_g.all_true, bool_g.ihalf);
        }
        // free function version
        {
            INFO("bitwise_not");
            CHECK_BATCH_EQ(bool_g.half, xsimd::bitwise_not(bool_g.ihalf));
        }
        {
            INFO("bitwise_or");
            CHECK_BATCH_EQ(xsimd::bitwise_or(bool_g.half, bool_g.ihalf), bool_g.all_true);
        }
        {
            INFO("bitwise_and");
            CHECK_BATCH_EQ(xsimd::bitwise_and(bool_g.half, bool_g.ihalf), bool_g.all_false);
        }
        {
            INFO("bitwise_xor");
            CHECK_BATCH_EQ(xsimd::bitwise_xor(bool_g.half, bool_g.all_true), bool_g.ihalf);
        }
    }

    void test_update_operations() const
    {
        auto bool_g = xsimd::get_bool<batch_bool_type> {};
        {
            auto tmp = bool_g.half;
            tmp |= bool_g.ihalf;
            bool res = xsimd::all(tmp);
            INFO("operator|=");
            CHECK_UNARY(res);
        }
        {
            auto tmp = bool_g.half;
            tmp &= bool_g.half;
            INFO("operator&=");
            CHECK_BATCH_EQ(tmp, bool_g.half);
        }
        {
            auto tmp = bool_g.half;
            tmp ^= bool_g.ihalf;
            bool res = xsimd::all(tmp);
            INFO("operator^=");
            CHECK_UNARY(res);
        }
    }

    void test_mask() const
    {
        auto bool_g = xsimd::get_bool<batch_bool_type> {};
        const uint64_t full_mask = ((uint64_t)-1) >> (64 - batch_bool_type::size);
        CHECK_EQ(bool_g.all_false.mask(), 0);
        CHECK_EQ(batch_bool_type::from_mask(bool_g.all_false.mask()).mask(), bool_g.all_false.mask());

        CHECK_EQ(bool_g.all_true.mask(), full_mask);
        CHECK_EQ(batch_bool_type::from_mask(bool_g.all_true.mask()).mask(), bool_g.all_true.mask());

        CHECK_EQ(bool_g.half.mask(), full_mask & ((uint64_t)-1) << (batch_bool_type::size / 2));
        CHECK_EQ(batch_bool_type::from_mask(bool_g.half.mask()).mask(), bool_g.half.mask());

        CHECK_EQ(bool_g.ihalf.mask(), full_mask & ~(((uint64_t)-1) << (batch_bool_type::size / 2)));
        CHECK_EQ(batch_bool_type::from_mask(bool_g.ihalf.mask()).mask(), bool_g.ihalf.mask());

        CHECK_EQ(bool_g.interspersed.mask(), full_mask & 0xAAAAAAAAAAAAAAAAul);
        CHECK_EQ(batch_bool_type::from_mask(bool_g.interspersed.mask()).mask(), bool_g.interspersed.mask());
    }

    void test_count() const
    {
        auto bool_g = xsimd::get_bool<batch_bool_type> {};
        CHECK_EQ(count(bool_g.all_false), 0);
        CHECK_EQ(count(bool_g.all_true), batch_bool_type::size);
        CHECK_EQ(count(bool_g.half), batch_bool_type::size / 2);
    }

    void test_comparison() const
    {
        auto bool_g = xsimd::get_bool<batch_bool_type> {};
        // eq
        {
            CHECK_BATCH_EQ(bool_g.half, !bool_g.ihalf);
            CHECK_BATCH_EQ(xsimd::eq(bool_g.half, !bool_g.ihalf), bool_g.all_true);
        }
        // neq
        {
            CHECK_BATCH_EQ(xsimd::neq(bool_g.half, bool_g.ihalf), bool_g.all_true);
        }
    }

    void test_mask_compile_time() const
    {
        xsimd_ct_mask_checker<T>::run();
    }

    void test_mask_runtime() const
    {
        auto bool_g = xsimd::get_bool<typename T::batch_bool_type> {};
        using batch_t = typename T::batch_bool_type;

        auto check_mask = [&](batch_t const& m, const char*)
        {
            constexpr std::size_t N = batch_t::size;
            // expected popcount via existing count()
            std::size_t expected_pop = xsimd::count(m);
            CHECK_EQ(m.popcount(), expected_pop);

            // expected is_all_zeros / is_all_ones
            CHECK_EQ(m.is_all_zeros(), expected_pop == 0);
            CHECK_EQ(m.is_all_ones(), expected_pop == N);

            // build a bool array to derive prefix/suffix and indices
            std::array<bool, N> bits;
            m.store_unaligned(bits.data());

            // countr_one (trailing ones from LSB)
            std::size_t cto = 0;
            for (std::size_t i = 0; i < N; ++i)
            {
                if (bits[i])
                    ++cto;
                else
                    break;
            }
            CHECK_EQ(m.countr_one(), cto);

            // countl_one (leading ones from MSB)
            std::size_t clo = 0;
            for (std::size_t i = 0; i < N; ++i)
            {
                std::size_t b = N - 1 - i;
                if (bits[b])
                    ++clo;
                else
                    break;
            }
            CHECK_EQ(m.countl_one(), clo);

            // first_one_index / last_one_index
            std::size_t first = N, last = N;
            for (std::size_t i = 0; i < N; ++i)
            {
                if (bits[i])
                {
                    first = i;
                    break;
                }
            }
            for (std::size_t i = 0; i < N; ++i)
            {
                std::size_t b = N - 1 - i;
                if (bits[b])
                {
                    last = b;
                    break;
                }
            }
            CHECK_EQ(m.first_one_index(), first);
            CHECK_EQ(m.last_one_index(), last);

            // prefix/suffix ones
            bool is_prefix = true;
            for (std::size_t i = 0; i < cto; ++i)
                if (!bits[i])
                    is_prefix = false;
            for (std::size_t i = cto; i < N; ++i)
                if (bits[i])
                    is_prefix = false;
            bool is_suffix = true;
            for (std::size_t i = 0; i < N - clo; ++i)
                if (bits[i])
                    is_suffix = false;
            for (std::size_t i = N - clo; i < N; ++i)
                if (!bits[i])
                    is_suffix = false;
            CHECK_EQ(m.is_prefix_ones(), is_prefix);
            CHECK_EQ(m.is_suffix_ones(), is_suffix);
        };

        check_mask(bool_g.all_false, "all_false");
        check_mask(bool_g.all_true, "all_true");
        check_mask(bool_g.half, "half");
        check_mask(bool_g.ihalf, "ihalf");
        check_mask(bool_g.interspersed, "interspersed");
    }

private:
    batch_type batch_lhs() const
    {
        return batch_type::load_unaligned(lhs.data());
    }

    batch_type batch_rhs() const
    {
        return batch_type::load_unaligned(rhs.data());
    }
};

TEST_CASE_TEMPLATE("[xsimd batch bool]", B, BATCH_TYPES)
{
    batch_bool_test<B> Test;

    SUBCASE("constructors") { Test.test_constructors(); }

    SUBCASE("load store") { Test.test_load_store(); }

    SUBCASE("any all") { Test.test_any_all(); }

    SUBCASE("logical operations") { Test.test_logical_operations(); }

    SUBCASE("bitwise operations") { Test.test_bitwise_operations(); }

    SUBCASE("update operations") { Test.test_update_operations(); }

    SUBCASE("mask") { Test.test_mask(); }

    SUBCASE("count") { Test.test_count(); }

    SUBCASE("eq neq") { Test.test_comparison(); }

    SUBCASE("mask utils (compile-time)") { Test.test_mask_compile_time(); }

    SUBCASE("mask utils (runtime)") { Test.test_mask_runtime(); }
}
#endif
