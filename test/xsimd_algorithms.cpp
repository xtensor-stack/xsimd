/***************************************************************************
* Copyright (c) 2016, Wolf Vollprecht, Johan Mabille and Sylvain Corlay    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <fstream>
#include <iostream>
#include <numeric>

#include "xsimd/xsimd.hpp"

#include "gtest/gtest.h"

struct binary_functor
{
    template <class T>
    T operator()(const T& a, const T& b) const
    {
        return a + b;
    }
};

struct unary_functor
{
    template <class T>
    T operator()(const T& a) const
    {
        return -a;
    }
};

TEST(xsimd, binary_transform)
{
    std::vector<double> expected(93);

    std::vector<double> a(93, 123), b(93, 123), c(93);
    std::vector<double, xsimd::aligned_allocator<double, XSIMD_DEFAULT_ALIGNMENT>> aa(93, 123), ba(93, 123), ca(93);

    std::transform(a.begin(), a.end(), b.begin(), expected.begin(),
                    binary_functor{});

    xsimd::transform(a.begin(), a.end(), b.begin(), c.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), c.begin()) && expected.size() == c.size());
    std::fill(c.begin(), c.end(), -1); // erase

    xsimd::transform(aa.begin(), aa.end(), ba.begin(), c.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), c.begin()) && expected.size() == c.size());
    std::fill(c.begin(), c.end(), -1); // erase

    xsimd::transform(aa.begin(), aa.end(), b.begin(), c.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), c.begin()) && expected.size() == c.size());
    std::fill(c.begin(), c.end(), -1); // erase

    xsimd::transform(a.begin(), a.end(), ba.begin(), c.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), c.begin()) && expected.size() == c.size());
    std::fill(c.begin(), c.end(), -1); // erase

    xsimd::transform(aa.begin(), aa.end(), ba.begin(), ca.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), ca.begin()) && expected.size() == ca.size());
    std::fill(ca.begin(), ca.end(), -1); // erase

    xsimd::transform(aa.begin(), aa.end(), b.begin(), ca.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), ca.begin()) && expected.size() == ca.size());
    std::fill(ca.begin(), ca.end(), -1); // erase

    xsimd::transform(a.begin(), a.end(), ba.begin(), ca.begin(),
                     binary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), ca.begin()) && expected.size() == ca.size());
    std::fill(ca.begin(), ca.end(), -1); // erase
}


TEST(xsimd, unary_transform)
{
    std::vector<double> expected(93);
    std::vector<double> a(93, 123), c(93);
    std::vector<double, xsimd::aligned_allocator<double, XSIMD_DEFAULT_ALIGNMENT>> aa(93, 123), ca(93);

    std::transform(a.begin(), a.end(), expected.begin(),
                   unary_functor{});

    xsimd::transform(a.begin(), a.end(), c.begin(),
                     unary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), c.begin()) && expected.size() == c.size());
    std::fill(c.begin(), c.end(), -1); // erase

    xsimd::transform(aa.begin(), aa.end(), c.begin(),
                     unary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), c.begin()) && expected.size() == c.size());
    std::fill(c.begin(), c.end(), -1); // erase

    xsimd::transform(a.begin(), a.end(), ca.begin(),
                     unary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), ca.begin()) && expected.size() == ca.size());
    std::fill(ca.begin(), ca.end(), -1); // erase

    xsimd::transform(aa.begin(), aa.end(), ca.begin(),
                     unary_functor{});
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), ca.begin()) && expected.size() == ca.size());
    std::fill(ca.begin(), ca.end(), -1); // erase
}



class xsimd_reduce : public ::testing::Test
{
public:
    using aligned_vec_t = std::vector<double, xsimd::aligned_allocator<double, XSIMD_DEFAULT_ALIGNMENT>>;

    static constexpr std::size_t num_elements = 4 * xsimd::simd_traits<double>::size;

    aligned_vec_t  vec = aligned_vec_t(num_elements, 123.);
    double         init = 1337.;

    struct multiply
    {
        template <class T>
        T operator()(const T& a, const T& b) const
        {
            return a * b;
        }
    };
};

TEST_F(xsimd_reduce, unaligned_begin_unaligned_end)
{
    auto const begin = std::next(vec.begin());
    auto const end = std::prev(vec.end());

    EXPECT_EQ(std::accumulate(begin, end, init), xsimd::reduce(begin, end, init));
}

TEST_F(xsimd_reduce, unaligned_begin_aligned_end)
{
    auto const begin = std::next(vec.begin());
    auto const end = vec.end();

    EXPECT_EQ(std::accumulate(begin, end, init), xsimd::reduce(begin, end, init));
}

TEST_F(xsimd_reduce, aligned_begin_unaligned_end)
{
    auto const begin = vec.begin();
    auto const end = std::prev(vec.end());

    EXPECT_EQ(std::accumulate(begin, end, init), xsimd::reduce(begin, end, init));
}

TEST_F(xsimd_reduce, aligned_begin_aligned_end)
{
    auto const begin = vec.begin();
    auto const end = vec.end();

    EXPECT_EQ(std::accumulate(begin, end, init), xsimd::reduce(begin, end, init));
}

TEST_F(xsimd_reduce, using_custom_binary_function)
{
    auto const begin = vec.begin();
    auto const end = vec.end();

    EXPECT_DOUBLE_EQ(std::accumulate(begin, end, init, multiply{}), xsimd::reduce(begin, end, init, multiply{}));
}

TEST(xsimd, iterator)
{
    std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> a(10 * 16, 0.2);
    std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> b(1000, 2.);
    std::vector<float, xsimd::aligned_allocator<float, XSIMD_DEFAULT_ALIGNMENT>> c(1000, 3.);

    std::iota(a.begin(), a.end(), 0.f);
    std::vector<float> a_cpy(a.begin(), a.end());

    using batch_type = typename xsimd::simd_traits<float>::type;
    auto begin = xsimd::aligned_iterator<batch_type>(&a[0]);
    auto end = xsimd::aligned_iterator<batch_type>(&a[0] + a.size());
 
    for (; begin != end; ++begin)
    {
        *begin = *begin / 2.f;
    }

    for (auto& el : a_cpy)
    {
        el /= 2.f;
    }

    for (auto& el : a_cpy) { std::cout << el << ", "; }
    for (auto& el : a) { std::cout << el << ", "; }
    EXPECT_TRUE(a.size() == a_cpy.size() && std::equal(a.begin(), a.end(), a_cpy.begin()));

    begin = xsimd::aligned_iterator<batch_type>(&a[0]);
    *begin = sin(*begin);

    for (std::size_t i = 0; i < batch_type::size; ++i)
    {
        EXPECT_NEAR(a[i], sinf(a_cpy[i]), 1e-6);
    }

#ifdef XSIMD_BATCH_DOUBLE_SIZE
    std::vector<std::complex<double>, xsimd::aligned_allocator<std::complex<double>, XSIMD_DEFAULT_ALIGNMENT>> ca(10 * 16, std::complex<double>(0.2));
    using cbatch_type = typename xsimd::simd_traits<std::complex<double>>::type;
    auto cbegin = xsimd::aligned_iterator<cbatch_type>(&ca[0]);
    auto cend = xsimd::aligned_iterator<cbatch_type>(&ca[0] + a.size());

    for (; cbegin != cend; ++cbegin)
    {
        *cbegin = (*cbegin + std::complex<double>(0, .3)) / 2.;
    }

    cbegin = xsimd::aligned_iterator<cbatch_type>(&ca[0]);
    *cbegin = sin(*cbegin);
    *cbegin = sqrt(*cbegin);
    auto real_part = abs(*(cbegin));
#endif

}