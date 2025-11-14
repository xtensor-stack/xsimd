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

#include <algorithm>
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

    struct mask_none
    {
        static constexpr bool get(std::size_t, std::size_t) noexcept { return false; }
    };

    struct mask_first
    {
        static constexpr bool get(std::size_t index, std::size_t) noexcept { return index == 0; }
    };

    struct mask_first_half
    {
        static constexpr bool get(std::size_t index, std::size_t size) noexcept { return index < (size / 2); }
    };

    struct mask_last_half
    {
        static constexpr bool get(std::size_t index, std::size_t size) noexcept { return index >= (size / 2); }
    };

    struct mask_first_n
    {
        static constexpr bool get(std::size_t index, std::size_t size) noexcept
        {
            return index < (size > 2 ? size / 3 : std::size_t(1));
        }
    };

    struct mask_last_n
    {
        static constexpr bool get(std::size_t index, std::size_t size) noexcept
        {
            return index >= size - (size > 2 ? size / 3 : std::size_t(1));
        }
    };

    struct mask_even
    {
        static constexpr bool get(std::size_t index, std::size_t) noexcept { return (index % 2) == 0; }
    };

    struct mask_odd
    {
        static constexpr bool get(std::size_t index, std::size_t) noexcept { return (index % 2) == 1; }
    };

    struct mask_pseudo_random
    {
        static constexpr bool get(std::size_t index, std::size_t size) noexcept
        {
            return ((index * 7) + 3) % size < (size > 2 ? size / 3 : std::size_t(1));
        }
    };

    struct mask_all
    {
        static constexpr bool get(std::size_t, std::size_t) noexcept { return true; }
    };

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

    void test_masked()
    {
        using arch = typename B::arch_type;
        using test_batch_type = xsimd::batch<float, arch>;
        constexpr std::size_t test_size = test_batch_type::size;
        using int_allocator_type = xsimd::default_allocator<int32_t, arch>;

        std::vector<int32_t, int_allocator_type> source(test_size);
        for (std::size_t i = 0; i < test_size; ++i)
        {
            source[i] = static_cast<int32_t>(i * 17 - 9);
        }

        struct cross_type_mask
        {
            static constexpr bool get(std::size_t index, std::size_t size) noexcept
            {
                return ((index & std::size_t(1)) != 0) || (size == std::size_t(1)) || ((index == size - std::size_t(1)) && ((size % std::size_t(2)) == 0));
            }
        };
        auto mask = xsimd::make_batch_bool_constant<typename test_batch_type::value_type, cross_type_mask, arch>();

        std::array<float, test_size> expected_load;
        expected_load.fill(0.f);
        for (std::size_t i = 0; i < test_size; ++i)
        {
            if (cross_type_mask::get(i, test_size))
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
            if (cross_type_mask::get(i, test_size))
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
        run_load_mask_pattern<mask_none>(v, name, b, expected, " masked none");
        run_load_mask_pattern<mask_first>(v, name, b, expected, " masked first element");
        run_load_mask_pattern<mask_first_half>(v, name, b, expected, " masked first half");
        run_load_mask_pattern<mask_last_half>(v, name, b, expected, " masked last half");
        run_load_mask_pattern<mask_first_n>(v, name, b, expected, " masked first N");
        run_load_mask_pattern<mask_last_n>(v, name, b, expected, " masked last N");
        run_load_mask_pattern<mask_even>(v, name, b, expected, " masked even elements");
        run_load_mask_pattern<mask_odd>(v, name, b, expected, " masked odd elements");
        run_load_mask_pattern<mask_pseudo_random>(v, name, b, expected, " masked pseudo random");
        run_load_mask_pattern<mask_all>(v, name, b, expected, " masked all elements");
    }

    template <class V>
    void run_mask_tests(const V&, const std::string&, batch_type&, const array_type&, std::false_type)
    {
    }

    template <class Generator, class V>
    void run_load_mask_pattern(const V& v, const std::string& name, batch_type& b, const array_type& expected, const std::string& label)
    {
        constexpr auto mask = xsimd::make_batch_bool_constant<value_type, Generator, typename batch_type::arch_type>();
        array_type expected_masked { 0 };

        for (std::size_t i = 0; i < size; ++i)
        {
            const bool active = Generator::get(i, size);
            expected_masked[i] = active ? expected[i] : value_type();
        }

        b = xsimd::load(v.data(), mask, xsimd::aligned_mode());
        INFO(name, label + " aligned");
        CHECK_BATCH_EQ(b, expected_masked);
        b = xsimd::load(v.data(), mask, xsimd::unaligned_mode());
        INFO(name, label + " unaligned");
        CHECK_BATCH_EQ(b, expected_masked);
    }

    template <class Generator, class V>
    void run_store_mask_pattern(const V& v, const std::string& name, batch_type& b, V& res, V& expected_masked, const std::string& label)
    {
        auto mask = xsimd::make_batch_bool_constant<value_type, Generator, typename batch_type::arch_type>();
        for (std::size_t i = 0; i < size; ++i)
        {
            expected_masked[i] = Generator::get(i, size) ? v[i] : value_type();
        }
        std::fill(res.begin(), res.end(), value_type());
        b.store(res.data(), mask, xsimd::aligned_mode());
        INFO(name, label + " aligned");
        CHECK_VECTOR_EQ(res, expected_masked);
        std::fill(res.begin(), res.end(), value_type());
        b.store(res.data(), mask, xsimd::unaligned_mode());
        INFO(name, label + " unaligned");
        CHECK_VECTOR_EQ(res, expected_masked);
    }

    template <class V>
    void run_store_mask_tests(const V& v, const std::string& name, batch_type& b, V& res, V& expected_masked, std::true_type)
    {
        run_store_mask_pattern<mask_first>(v, name, b, res, expected_masked, " masked first element");
        run_store_mask_pattern<mask_first_half>(v, name, b, res, expected_masked, " masked first half");
        run_store_mask_pattern<mask_last_half>(v, name, b, res, expected_masked, " masked last half");
        run_store_mask_pattern<mask_first_n>(v, name, b, res, expected_masked, " masked first N");
        run_store_mask_pattern<mask_last_n>(v, name, b, res, expected_masked, " masked last N");
        run_store_mask_pattern<mask_even>(v, name, b, res, expected_masked, " masked even elements");
        run_store_mask_pattern<mask_odd>(v, name, b, res, expected_masked, " masked odd elements");
        run_store_mask_pattern<mask_pseudo_random>(v, name, b, res, expected_masked, " masked pseudo random");
        run_store_mask_pattern<mask_all>(v, name, b, res, expected_masked, " masked all elements");
    }

    template <class V>
    void run_store_mask_tests(const V&, const std::string&, batch_type&, V&, V&, std::false_type)
    {
    }

    template <class V>
    void run_store_mask_section(const V& v,
                                const std::string& name,
                                batch_type& b,
                                V& res,
                                V& expected_masked,
                                std::true_type)
    {
        static constexpr auto sentinel = static_cast<value_type>(37);
        V sentinel_expected(size, sentinel);

        auto zero_mask = xsimd::make_batch_bool_constant<value_type, mask_none, typename batch_type::arch_type>();
        std::fill(res.begin(), res.end(), sentinel);
        b.store(res.data(), zero_mask, xsimd::aligned_mode());
        INFO(name, " masked none aligned store");
        CHECK_VECTOR_EQ(res, sentinel_expected);

        V scratch(res.size() + size);
        std::fill(scratch.begin(), scratch.end(), sentinel);
        auto* scratch_ptr = scratch.data() + 1;
        b.store(scratch_ptr, zero_mask, xsimd::unaligned_mode());
        INFO(name, " masked none unaligned store");

        V scratch_slice(res.size());
        std::copy(scratch_ptr, scratch_ptr + scratch_slice.size(), scratch_slice.begin());
        CHECK_VECTOR_EQ(scratch_slice, sentinel_expected);
        CHECK(std::all_of(scratch.begin(), scratch.end(), [](const value_type v)
                          { return v == sentinel; }));

        run_store_mask_tests(v, name, b, res, expected_masked, std::true_type {});
    }

    template <class V>
    void run_store_mask_section(const V&, const std::string&, batch_type&, V&, V&, std::false_type)
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

        V expected_masked(size);

        run_store_mask_section(v, name, b, res, expected_masked, std::is_same<typename V::value_type, value_type> {});
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

    SUBCASE("masked") { Test.test_masked(); }
}
#endif
