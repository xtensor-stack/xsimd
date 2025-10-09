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
#include <random>
#include <type_traits>

#include "test_utils.hpp"

template <class B>
struct load_store_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    using index_type = typename xsimd::as_integer_t<batch_type>;
    using batch_bool_type = typename batch_type::batch_bool_type;
    template <class T>
    using allocator = xsimd::default_allocator<T, typename B::arch_type>;
    static constexpr size_t size = B::size;
    using array_type = std::array<value_type, size>;
    using int8_vector_type = std::vector<int8_t, allocator<int8_t>>;
    using uint8_vector_type = std::vector<uint8_t, allocator<uint8_t>>;
    using int16_vector_type = std::vector<int16_t, allocator<int16_t>>;
    using uint16_vector_type = std::vector<uint16_t, allocator<uint16_t>>;
    using int32_vector_type = std::vector<int32_t, allocator<int32_t>>;
    using uint32_vector_type = std::vector<uint32_t, allocator<uint32_t>>;
    using int64_vector_type = std::vector<int64_t, allocator<int64_t>>;
    using uint64_vector_type = std::vector<uint64_t, allocator<uint64_t>>;
#ifdef XSIMD_32_BIT_ABI
    using long_vector_type = std::vector<long, allocator<long>>;
    using ulong_vector_type = std::vector<unsigned long, allocator<unsigned long>>;
#endif
    using float_vector_type = std::vector<float, allocator<float>>;
#if !XSIMD_WITH_NEON || XSIMD_WITH_NEON64
    using double_vector_type = std::vector<double, allocator<double>>;
#endif

    int8_vector_type i8_vec;
    uint8_vector_type ui8_vec;
    int16_vector_type i16_vec;
    uint16_vector_type ui16_vec;
    int32_vector_type i32_vec;
    uint32_vector_type ui32_vec;
    int64_vector_type i64_vec;
    uint64_vector_type ui64_vec;
#ifdef XSIMD_32_BIT_ABI
    long_vector_type l_vec;
    ulong_vector_type ul_vec;
#endif
    float_vector_type f_vec;
#if !XSIMD_WITH_NEON || XSIMD_WITH_NEON64
    double_vector_type d_vec;
#endif

    array_type expected;

    load_store_test()
    {
        init_test_vector(i8_vec);
        init_test_vector(ui8_vec);
        init_test_vector(i16_vec);
        init_test_vector(ui16_vec);
        init_test_vector(i32_vec);
        init_test_vector(ui32_vec);
        init_test_vector(i64_vec);
        init_test_vector(ui64_vec);
#ifdef XSIMD_32_BIT_ABI
        init_test_vector(l_vec);
        init_test_vector(ul_vec);
#endif
        init_test_vector(f_vec);
#if !XSIMD_WITH_NEON || XSIMD_WITH_NEON64
        init_test_vector(d_vec);
#endif
    }

    void test_load()
    {
        test_load_impl(i8_vec, "load int8_t");
        test_load_impl(ui8_vec, "load uint8_t");
        test_load_impl(i16_vec, "load int16_t");
        test_load_impl(ui16_vec, "load uint16_t");
        test_load_impl(i32_vec, "load int32_t");
        test_load_impl(ui32_vec, "load uint32_t");
        test_load_impl(i64_vec, "load int64_t");
        test_load_impl(ui64_vec, "load uint64_t");
#ifdef XSIMD_32_BIT_ABI
        test_load_impl(l_vec, "load long");
        test_load_impl(ul_vec, "load unsigned long");
#endif
        test_load_impl(f_vec, "load float");
#if !XSIMD_WITH_NEON || XSIMD_WITH_NEON64
        test_load_impl(d_vec, "load double");
#endif
    }

    void test_store()
    {
        test_store_impl(i8_vec, "load int8_t");
        test_store_impl(ui8_vec, "load uint8_t");
        test_store_impl(i16_vec, "load int16_t");
        test_store_impl(ui16_vec, "load uint16_t");
        test_store_impl(i32_vec, "load int32_t");
        test_store_impl(ui32_vec, "load uint32_t");
        test_store_impl(i64_vec, "load int64_t");
        test_store_impl(ui64_vec, "load uint64_t");
#ifdef XSIMD_32_BIT_ABI
        test_store_impl(l_vec, "load long");
        test_store_impl(ul_vec, "load unsigned long");
#endif
        test_store_impl(f_vec, "load float");
#if !XSIMD_WITH_NEON || XSIMD_WITH_NEON64
        test_store_impl(d_vec, "load double");
#endif
    }
    void test_gather()
    {
        test_gather_impl(i8_vec, "gather int8_t");
        test_gather_impl(ui8_vec, "gather uint8_t");
        test_gather_impl(i16_vec, "gather int16_t");
        test_gather_impl(ui16_vec, "gather uint16_t");
        test_gather_impl(i32_vec, "gather int32_t");
        test_gather_impl(ui32_vec, "gather uint32_t");
        test_gather_impl(i64_vec, "gather int64_t");
        test_gather_impl(ui64_vec, "gather uint64_t");
#ifdef XSIMD_32_BIT_ABI
        test_gather_impl(l_vec, "gather long");
        test_gather_impl(ul_vec, "gather unsigned long");
#endif
        test_gather_impl(f_vec, "gather float");
        test_gather_impl(f_vec, "gather float");
#if !XSIMD_WITH_NEON || XSIMD_WITH_NEON64
        test_gather_impl(d_vec, "gather double");
#endif
    }

    void test_scatter()
    {
        test_scatter_impl(i8_vec, "scatter int8_t");
        test_scatter_impl(ui8_vec, "scatter uint8_t");
        test_scatter_impl(i16_vec, "scatter int16_t");
        test_scatter_impl(ui16_vec, "scatter uint16_t");
        test_scatter_impl(i32_vec, "scatter int32_t");
        test_scatter_impl(ui32_vec, "scatter uint32_t");
        test_scatter_impl(i64_vec, "scatter int64_t");
        test_scatter_impl(ui64_vec, "scatter uint64_t");
#ifdef XSIMD_32_BIT_ABI
        test_scatter_impl(l_vec, "scatter long");
        test_scatter_impl(ul_vec, "scatter unsigned long");
#endif
        test_scatter_impl(f_vec, "scatter float");
#if !XSIMD_WITH_NEON || XSIMD_WITH_NEON64
        test_scatter_impl(d_vec, "scatter double");
#endif
    }

    void test_masked_cross_type()
    {
        using arch = typename B::arch_type;
        using test_batch_type = xsimd::batch<float, arch>;
        using bool_batch = typename test_batch_type::batch_bool_type;
        constexpr std::size_t test_size = test_batch_type::size;
        using int_allocator_type = xsimd::default_allocator<int32_t, arch>;

        std::vector<int32_t, int_allocator_type> source(test_size);
        for (std::size_t i = 0; i < test_size; ++i)
        {
            source[i] = static_cast<int32_t>(i * 17 - 9);
        }

        bool mask_init[test_size];
        for (std::size_t i = 0; i < test_size; ++i)
        {
            const bool odd_lane = (i & 1) == 1;
            const bool include_last = (test_size == 1) ? true : (i == test_size - 1 && (test_size % 2 == 0));
            mask_init[i] = odd_lane || include_last;
        }
        auto mask = bool_batch::load_unaligned(mask_init);

        std::array<float, test_size> expected_load;
        expected_load.fill(0.f);
        for (std::size_t i = 0; i < test_size; ++i)
        {
            if (mask_init[i])
            {
                expected_load[i] = static_cast<float>(source[i]);
            }
        }

        auto loaded_aligned = test_batch_type::load(source.data(), mask, xsimd::aligned_mode());
        INFO("cross-type masked load aligned");
        CHECK_BATCH_EQ(loaded_aligned, expected_load);

        auto loaded_unaligned = test_batch_type::load(source.data(), mask, xsimd::unaligned_mode());
        INFO("cross-type masked load unaligned");
        CHECK_BATCH_EQ(loaded_unaligned, expected_load);

        std::array<float, test_size> values;
        for (std::size_t i = 0; i < test_size; ++i)
        {
            values[i] = static_cast<float>(static_cast<int>(i) * 2 - 7) / 3.f;
        }
        auto value_batch = test_batch_type::load_unaligned(values.data());

        std::vector<int32_t, int_allocator_type> destination(test_size, -19);
        std::vector<int32_t, int_allocator_type> expected_store(test_size, -19);
        for (std::size_t i = 0; i < test_size; ++i)
        {
            if (mask_init[i])
            {
                expected_store[i] = static_cast<int32_t>(values[i]);
            }
        }

        value_batch.store(destination.data(), mask, xsimd::aligned_mode());
        INFO("cross-type masked store aligned");
        CHECK_VECTOR_EQ(destination, expected_store);

        std::fill(destination.begin(), destination.end(), -19);
        value_batch.store(destination.data(), mask, xsimd::unaligned_mode());
        INFO("cross-type masked store unaligned");
        CHECK_VECTOR_EQ(destination, expected_store);
    }

private:
#ifdef XSIMD_WITH_SSE2
    struct test_load_as_return_type
    {
        using lower_arch = xsimd::sse2;
        using expected_batch_type = xsimd::batch<float, lower_arch>;
        using load_as_return_type = decltype(xsimd::load_as<float, lower_arch>(std::declval<float*>(), xsimd::aligned_mode()));
        static_assert(std::is_same<load_as_return_type, expected_batch_type>::value, "honoring arch parameter");
    };
#endif

    template <class V>
    void test_load_impl(const V& v, const std::string& name)
    {
        std::copy(v.cbegin(), v.cend(), expected.begin());

        batch_type b = batch_type::load_unaligned(v.data());
        INFO(name, " unaligned");
        CHECK_BATCH_EQ(b, expected);

        b = batch_type::load_aligned(v.data());
        INFO(name, " aligned");
        CHECK_BATCH_EQ(b, expected);

        b = xsimd::load_as<value_type>(v.data(), xsimd::unaligned_mode());
        INFO(name, " unaligned (load_as)");
        CHECK_BATCH_EQ(b, expected);

        b = xsimd::load_as<value_type>(v.data(), xsimd::aligned_mode());
        INFO(name, " aligned (load_as)");
        CHECK_BATCH_EQ(b, expected);

        run_mask_tests(v, name, b, expected, std::is_same<typename V::value_type, value_type> {});
    }

    template <class V>
    void run_mask_tests(const V& v, const std::string& name, batch_type& b, const array_type& expected, std::true_type)
    {
        bool mask_init[size];
        array_type expected_masked;

        std::fill(std::begin(mask_init), std::end(mask_init), false);
        auto zero_mask = batch_bool_type::load_unaligned(mask_init);
        expected_masked.fill(value_type());
        b = xsimd::load(v.data(), zero_mask, xsimd::aligned_mode());
        INFO(name, " masked none aligned");
        CHECK_BATCH_EQ(b, expected_masked);
        b = xsimd::load(v.data() + v.size(), zero_mask, xsimd::unaligned_mode());
        INFO(name, " masked none unaligned");
        CHECK_BATCH_EQ(b, expected_masked);
        CHECK(zero_mask.none());
        CHECK_FALSE(zero_mask.any());
        CHECK_FALSE(zero_mask.all());
        CHECK_EQ(zero_mask.countr_zero(), size);
        CHECK_EQ(zero_mask.countl_zero(), size);
        uint64_t zero_bits = zero_mask.truncated_mask();
        CHECK_EQ(zero_bits, uint64_t(0));
        std::vector<std::size_t> zero_indices;
        for (std::size_t idx = 0; idx < size; ++idx)
        {
            if (zero_mask.get(idx))
            {
                zero_indices.push_back(idx);
            }
        }
        CHECK(zero_indices.empty());

        auto check_mask = [&](const std::function<bool(size_t)>& predicate, const std::string& label)
        {
            for (size_t i = 0; i < size; ++i)
            {
                mask_init[i] = predicate(i);
            }
            auto mask = batch_bool_type::load_unaligned(mask_init);
            expected_masked.fill(value_type());
            for (size_t i = 0; i < size; ++i)
            {
                if (mask_init[i])
                {
                    expected_masked[i] = expected[i];
                }
            }
            std::size_t expected_pop = 0;
            std::size_t expected_countr_one = 0;
            std::size_t expected_countl_one = 0;
            std::size_t expected_countr_zero = 0;
            std::size_t expected_countl_zero = 0;
            uint64_t expected_bits = 0;
            for (size_t i = 0; i < size; ++i)
            {
                if (mask_init[i])
                {
                    expected_bits |= (uint64_t(1) << i);
                    ++expected_pop;
                }
            }
            // count trailing zeros
            while (expected_countr_zero < size && !mask_init[expected_countr_zero])
            {
                ++expected_countr_zero;
            }
            while (expected_countl_zero < size && !mask_init[size - 1 - expected_countl_zero])
            {
                ++expected_countl_zero;
            }
            // adjust trailing ones count (stop once first zero encountered)
            expected_countr_one = 0;
            for (size_t i = 0; i < size; ++i)
            {
                if (mask_init[i])
                {
                    ++expected_countr_one;
                }
                else
                {
                    break;
                }
            }
            expected_countl_one = 0;
            for (size_t i = 0; i < size; ++i)
            {
                size_t idx = size - 1 - i;
                if (mask_init[idx])
                {
                    ++expected_countl_one;
                }
                else
                {
                    break;
                }
            }

            b = xsimd::load(v.data(), mask, xsimd::aligned_mode());
            INFO(name, label + " aligned");
            CHECK_BATCH_EQ(b, expected_masked);
            b = xsimd::load(v.data(), mask, xsimd::unaligned_mode());
            INFO(name, label + " unaligned");
            CHECK_BATCH_EQ(b, expected_masked);

            CHECK_EQ(mask.truncated_mask(), expected_bits);
            CHECK_EQ(mask.popcount(), expected_pop);
            CHECK_EQ(mask.none(), expected_pop == 0);
            CHECK_EQ(mask.any(), expected_pop != 0);
            CHECK_EQ(mask.all(), expected_pop == size);
            CHECK_EQ(mask.countr_one(), expected_countr_one);
            CHECK_EQ(mask.countl_one(), expected_countl_one);
            CHECK_EQ(mask.countr_zero(), expected_countr_zero);
            CHECK_EQ(mask.countl_zero(), expected_countl_zero);

            std::vector<std::size_t> indices;
            for (std::size_t idx = 0; idx < size; ++idx)
            {
                if (mask.get(idx))
                {
                    indices.push_back(idx);
                }
            }
            std::vector<std::size_t> expected_indices;
            for (size_t i = 0; i < size; ++i)
            {
                if (mask_init[i])
                {
                    expected_indices.push_back(i);
                }
            }
            CHECK_EQ(indices, expected_indices);
        };

        check_mask([](size_t i)
                   { return i == 0; },
                   " masked first element");
        check_mask([](size_t i)
                   { return i < size / 2; },
                   " masked first half");
        check_mask([](size_t i)
                   { return i >= size / 2; },
                   " masked last half");
        const size_t n = size > 2 ? size / 3 : 1;
        check_mask([&](size_t i)
                   { return i < n; },
                   " masked first N");
        check_mask([&](size_t i)
                   { return i >= size - n; },
                   " masked last N");
        check_mask([](size_t i)
                   { return i % 2 == 0; },
                   " masked even elements");
        check_mask([](size_t i)
                   { return i % 2 == 1; },
                   " masked odd elements");
        check_mask([&](size_t i)
                   { return ((i * 7) + 3) % size < n; },
                   " masked pseudo random");
        check_mask([](size_t)
                   { return true; },
                   " masked all elements");
    }

    template <class V>
    void run_mask_tests(const V&, const std::string&, batch_type&, const array_type&, std::false_type)
    {
    }

    struct test_load_char
    {
        /* Make sure xsimd doesn't try to be smart with char types */
        static_assert(std::is_same<xsimd::batch<char>, decltype(xsimd::load_as<char>(std::declval<char*>(), xsimd::aligned_mode()))>::value,
                      "honor explicit type request");
        static_assert(std::is_same<xsimd::batch<unsigned char>, decltype(xsimd::load_as<unsigned char>(std::declval<unsigned char*>(), xsimd::aligned_mode()))>::value,
                      "honor explicit type request");
        static_assert(std::is_same<xsimd::batch<signed char>, decltype(xsimd::load_as<signed char>(std::declval<signed char*>(), xsimd::aligned_mode()))>::value,
                      "honor explicit type request");
    };

    template <class V>
    void test_store_impl(const V& v, const std::string& name)
    {
        batch_type b = batch_type::load_aligned(v.data());
        V res(size);

        b.store_unaligned(res.data());
        INFO(name, " unaligned");
        CHECK_VECTOR_EQ(res, v);

        b.store_aligned(res.data());
        INFO(name, " aligned");
        CHECK_VECTOR_EQ(res, v);

        xsimd::store_as(res.data(), b, xsimd::unaligned_mode());
        INFO(name, " unaligned (store_as)");
        CHECK_VECTOR_EQ(res, v);

        xsimd::store_as(res.data(), b, xsimd::aligned_mode());
        INFO(name, " aligned (store_as)");
        CHECK_VECTOR_EQ(res, v);

        bool mask_init[size];
        V expected_masked(size);
        const auto sentinel = static_cast<value_type>(37);
        V sentinel_expected(size, sentinel);

        std::fill(mask_init, mask_init + size, false);
        auto zero_mask = batch_bool_type::load_unaligned(mask_init);
        std::fill(res.begin(), res.end(), sentinel);
        b.store(res.data(), zero_mask, xsimd::aligned_mode());
        INFO(name, " masked none aligned store");
        CHECK_VECTOR_EQ(res, sentinel_expected);
        b.store(res.data() + res.size(), zero_mask, xsimd::unaligned_mode());
        INFO(name, " masked none unaligned store");
        CHECK_VECTOR_EQ(res, sentinel_expected);

        const auto check_mask = [&](const std::function<bool(size_t)>& predicate, const std::string& label)
        {
            for (size_t i = 0; i < size; ++i)
            {
                mask_init[i] = predicate(i);
                expected_masked[i] = mask_init[i] ? v[i] : value_type();
            }
            auto mask = batch_bool_type::load_unaligned(mask_init);
            std::fill(res.begin(), res.end(), value_type());
            b.store(res.data(), mask, xsimd::aligned_mode());
            INFO(name, label + " aligned");
            CHECK_VECTOR_EQ(res, expected_masked);
            std::fill(res.begin(), res.end(), value_type());
            b.store(res.data(), mask, xsimd::unaligned_mode());
            INFO(name, label + " unaligned");
            CHECK_VECTOR_EQ(res, expected_masked);
        };

        check_mask([](size_t i)
                   { return i == 0; },
                   " masked first element");
        check_mask([](size_t i)
                   { return i < size / 2; },
                   " masked first half");
        check_mask([](size_t i)
                   { return i >= size / 2; },
                   " masked last half");
        const size_t n = size > 2 ? size / 3 : 1;
        check_mask([&](size_t i)
                   { return i < n; },
                   " masked first N");
        check_mask([&](size_t i)
                   { return i >= size - n; },
                   " masked last N");
        check_mask([](size_t i)
                   { return i % 2 == 0; },
                   " masked even elements");
        check_mask([](size_t i)
                   { return i % 2 == 1; },
                   " masked odd elements");
        check_mask([&](size_t i)
                   { return ((i * 7) + 3) % size < n; },
                   " masked pseudo random");
    }

    template <class V>
    void test_gather_impl(const V& v, const std::string& name)
    {
        std::copy(v.cbegin(), v.cend(), expected.begin());
        index_type index = xsimd::detail::make_sequence_as_batch<index_type>();
        batch_type b = batch_type::gather(v.data(), index);
        INFO(name, " (in order)");
        CHECK_BATCH_EQ(b, expected);

        std::reverse_copy(v.cbegin(), v.cend(), expected.begin());
        std::array<typename index_type::value_type, index_type::size> index_reverse;
        index.store_unaligned(index_reverse.data());
        std::reverse(index_reverse.begin(), index_reverse.end());
        index = index_type::load_unaligned(index_reverse.data());
        b = batch_type::gather(v.data(), index);
        INFO(name, " (in reverse order)");
        CHECK_BATCH_EQ(b, expected);
    }

    template <class V>
    void test_scatter_impl(const V& v, const std::string& name)
    {
        batch_type b = batch_type::load_aligned(v.data());
        index_type index = xsimd::detail::make_sequence_as_batch<index_type>();
        V res(size);

        b.scatter(res.data(), index);
        INFO(name, " (in order)");
        CHECK_VECTOR_EQ(res, v);

        V reverse_v(size);
        std::reverse_copy(v.cbegin(), v.cend(), reverse_v.begin());
        std::array<typename index_type::value_type, index_type::size> reverse_index;
        index.store_unaligned(reverse_index.data());
        std::reverse(reverse_index.begin(), reverse_index.end());
        index = index_type::load_unaligned(reverse_index.data());
        b.scatter(res.data(), index);
        INFO(name, " (in reverse order)");
        CHECK_VECTOR_EQ(res, reverse_v);
    }

    template <class V>
    void init_test_vector(V& vec)
    {
        vec.resize(size);

        int min = 0;
        int max = 100;

        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(min, max);

        auto gen = [&distribution, &generator]()
        {
            return static_cast<value_type>(distribution(generator));
        };

        std::generate(vec.begin(), vec.end(), gen);
    }
};

template <class B>
constexpr size_t load_store_test<B>::size;

TEST_CASE_TEMPLATE("[load store]", B, BATCH_TYPES)
{
    load_store_test<B> Test;
    SUBCASE("load") { Test.test_load(); }

    SUBCASE("store") { Test.test_store(); }

    SUBCASE("gather") { Test.test_gather(); }

    SUBCASE("scatter") { Test.test_scatter(); }

    SUBCASE("masked cross type") { Test.test_masked_cross_type(); }
}
#endif
