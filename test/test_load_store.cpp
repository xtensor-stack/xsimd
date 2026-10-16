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

#include "test_utils.hpp"

#include <algorithm>
#include <random>
#include <type_traits>

// Anti-rot guard for the internal has_mask_load_v / has_mask_store_v trait. It is
// a pure type-level computation (no batch instantiation), and every arch tag is
// always declared, so the value is build-independent -- the whole (size x arch)
// matrix is asserted unconditionally, no XSIMD_WITH_* guards. Capability is per
// size and load == store, so one helper checks both ops at all four sizes
// (float/double track int32/int64). Each line mirrors a row of the matrix.
namespace test_has_mask_trait
{
    using xsimd::batch;
    using xsimd::detail::has_mask_load_v;
    using xsimd::detail::has_mask_store_v;

    template <class A>
    constexpr bool caps(bool b1, bool b2, bool b4, bool b8)
    {
        return has_mask_load_v<batch<int8_t, A>> == b1 && has_mask_store_v<batch<int8_t, A>> == b1
            && has_mask_load_v<batch<int16_t, A>> == b2 && has_mask_store_v<batch<int16_t, A>> == b2
            && has_mask_load_v<batch<int32_t, A>> == b4 && has_mask_store_v<batch<int32_t, A>> == b4
            && has_mask_load_v<batch<int64_t, A>> == b8 && has_mask_store_v<batch<int64_t, A>> == b8;
    }

    //                                            i8     i16    i32    i64
    static_assert(caps<xsimd::sse2>(false, false, false, false), "sse2: scalar fallback");
    static_assert(caps<xsimd::sse4_2>(false, false, false, false), "sse4.2: scalar fallback");
    static_assert(caps<xsimd::avx>(false, false, true, true), "avx: 4/8 via vmaskmov");
    static_assert(caps<xsimd::avx_128>(false, false, true, true), "avx_128: 4/8 via vmaskmov");
    static_assert(caps<xsimd::avx2>(false, false, true, true), "avx2: 4/8 native");
    static_assert(caps<xsimd::fma3<xsimd::avx2>>(false, false, true, true), "fma3<avx2> follows avx2");
    static_assert(caps<xsimd::avx512f>(false, false, true, true), "avx512f: 4/8");
    static_assert(caps<xsimd::avx512bw>(true, true, true, true), "avx512bw: all sizes");
    static_assert(caps<xsimd::neon>(false, false, false, false), "neon: scalar fallback");
    static_assert(caps<xsimd::neon64>(false, false, false, false), "neon64: scalar fallback");
    static_assert(caps<xsimd::sve>(true, true, true, true), "sve: predicate-native");
    static_assert(caps<xsimd::rvv>(true, true, true, true), "rvv: predicate-native");
}

template <class B>
struct load_store_test
{
    using batch_type = B;
    using value_type = typename B::value_type;
    using index_type = typename xsimd::as_integer_t<batch_type>;
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
        static_assert(std::is_same_v<load_as_return_type, expected_batch_type>, "honoring arch parameter");
    };
#endif

    template <class Ptr>
    void stream_load_if_same(Ptr const* ptr, batch_type& b, array_type const& expected_values, const std::string& name,
                             std::true_type) const
    {
        b = xsimd::load(ptr, xsimd::stream_mode());
        INFO(name, " stream (load)");
        CHECK_BATCH_EQ(b, expected_values);
    }

    template <class Ptr>
    void stream_load_if_same(Ptr const*, batch_type&, array_type const&, const std::string&, std::false_type) const
    {
    }

    template <class Vec>
    void stream_store_if_same(Vec& res, batch_type const& b, Vec const& reference, const std::string& name, std::true_type) const
    {
        xsimd::store(res.data(), b, xsimd::stream_mode());
        INFO(name, " stream (store)");
        CHECK_VECTOR_EQ(res, reference);
    }

    template <class Vec>
    void stream_store_if_same(Vec&, batch_type const&, Vec const&, const std::string&, std::false_type) const
    {
    }

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

        b = batch_type::load(v.data(), xsimd::stream_mode());
        INFO(name, " stream (batch::load)");
        CHECK_BATCH_EQ(b, expected);

        b = xsimd::load_as<value_type>(v.data(), xsimd::unaligned_mode());
        INFO(name, " unaligned (load_as)");
        CHECK_BATCH_EQ(b, expected);

        b = xsimd::load_as<value_type>(v.data(), xsimd::aligned_mode());
        INFO(name, " aligned (load_as)");
        CHECK_BATCH_EQ(b, expected);

        b = xsimd::load_as<value_type>(v.data(), xsimd::stream_mode());
        INFO(name, " stream (load_as)");
        CHECK_BATCH_EQ(b, expected);

        stream_load_if_same(v.data(), b, expected, name,
                            std::integral_constant<bool, std::is_same_v<typename V::value_type, value_type>> {});
    }

    struct test_load_char
    {
        /* Make sure xsimd doesn't try to be smart with char types */
        static_assert(std::is_same_v<xsimd::batch<char>, decltype(xsimd::load_as<char>(std::declval<char*>(), xsimd::aligned_mode()))>,
                      "honor explicit type request");
        static_assert(std::is_same_v<xsimd::batch<unsigned char>, decltype(xsimd::load_as<unsigned char>(std::declval<unsigned char*>(), xsimd::aligned_mode()))>,
                      "honor explicit type request");
        static_assert(std::is_same_v<xsimd::batch<signed char>, decltype(xsimd::load_as<signed char>(std::declval<signed char*>(), xsimd::aligned_mode()))>,
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

        b.store(res.data(), xsimd::stream_mode());
        INFO(name, " stream (batch::store)");
        CHECK_VECTOR_EQ(res, v);

        xsimd::store_as(res.data(), b, xsimd::stream_mode());
        INFO(name, " stream (store_as)");
        CHECK_VECTOR_EQ(res, v);

        stream_store_if_same(res, b, v, name,
                             std::integral_constant<bool, std::is_same_v<typename V::value_type, value_type>> {});
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

template <class B>
struct masked_load_store_test
{
    using value_type = typename B::value_type;
    using arch_type = typename B::arch_type;
    using mask_type = typename B::batch_bool_type;
    using mask_value_type = typename mask_type::operand_type;
    static constexpr std::size_t size = B::size;
    using storage_type = std::array<value_type, size + 2>;

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

    struct mask_all_but_last
    {
        static constexpr bool get(std::size_t index, std::size_t size) noexcept { return index + 1 < size; }
    };

    struct mask_all_but_first
    {
        static constexpr bool get(std::size_t index, std::size_t) noexcept { return index != 0; }
    };

    struct mask_last
    {
        static constexpr bool get(std::size_t index, std::size_t size) noexcept { return index + 1 == size; }
    };

    static value_type make_value(int i)
    {
        if constexpr (xsimd::is_batch_complex<B>::value)
            return value_type(mask_value_type(i + 1), mask_value_type(-2 * i - 3));
        else
            return value_type(i + 1);
    }

    template <class Generator, class Mask, class Mode>
    void check(Mask mask, Mode mode, std::size_t offset) const
    {
        INFO("offset = ", offset, ", runtime mask = ", xsimd::is_batch_bool<Mask>::value);
        alignas(arch_type::alignment()) storage_type input;
        alignas(arch_type::alignment()) storage_type output;
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = make_value(static_cast<int>(i));

        const value_type sentinel = make_value(100);
        const B source = B::load_unaligned(input.data() + offset);
        auto check_results = [&](const auto& loaded)
        {
            static_assert(std::is_same_v<std::decay_t<decltype(loaded)>, B>,
                          "masked load preserves batch type");
            for (std::size_t i = 0; i < size; ++i)
            {
                const value_type expected = Generator::get(i, size) ? input[i + offset] : value_type {};
                CHECK_EQ(loaded.get(i), expected);
            }
            for (std::size_t i = 0; i < output.size(); ++i)
            {
                const bool active = i >= offset && i < offset + size && Generator::get(i - offset, size);
                CHECK_EQ(output[i], active ? input[i] : sentinel);
            }
        };

        output.fill(sentinel);
        xsimd::store(output.data() + offset, source, mask, mode);
        check_results(xsimd::load(input.data() + offset, mask, mode));

        if (offset == 0)
        {
            INFO("default mode");
            output.fill(sentinel);
            xsimd::store(output.data(), source, mask);
            check_results(xsimd::load(input.data(), mask));
        }
    }

    template <class Generator>
    void check_pattern(const char* name) const
    {
        INFO(std::string(name));
        constexpr auto constant_mask = xsimd::make_batch_bool_constant<mask_value_type, Generator, arch_type>();
        const auto runtime_mask = constant_mask.as_batch_bool();
        check<Generator>(constant_mask, xsimd::aligned_mode {}, 0);
        check<Generator>(constant_mask, xsimd::unaligned_mode {}, 1);
        check<Generator>(runtime_mask, xsimd::aligned_mode {}, 0);
        check<Generator>(runtime_mask, xsimd::unaligned_mode {}, 1);
    }

    void run() const
    {
        check_pattern<mask_none>("none");
        check_pattern<mask_all>("all");
        check_pattern<mask_first>("first element");
        check_pattern<mask_last>("last element");
        check_pattern<mask_first_half>("first half");
        check_pattern<mask_last_half>("last half");
        check_pattern<mask_even>("even elements");
        check_pattern<mask_odd>("odd elements");
        check_pattern<mask_all_but_first>("all but first");
        check_pattern<mask_all_but_last>("all but last");
        check_pattern<mask_pseudo_random>("pseudo random");
    }
};

TEST_CASE_TEMPLATE("[masked load store]", B, BATCH_TYPES, BATCH_COMPLEX_TYPES)
{
    masked_load_store_test<B> {}.run();
}

#endif
