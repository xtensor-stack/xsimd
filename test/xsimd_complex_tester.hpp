/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_COMPLEX_TESTER_HPP
#define XSIMD_COMPLEX_TESTER_HPP

#include <complex>
#include <cstddef>
#include <vector>

#include "xsimd/memory/xsimd_aligned_allocator.hpp"

namespace xsimd
{
    template <class T, std::size_t N>
    class batch;

    template <class T, std::size_t N, std::size_t A>
    struct simd_complex_tester
    {
        using vector_type = batch<T, N>;
        using value_type = T;
        using res_type = std::vector<T, aligned_allocator<value_type, A>>;
        using real_value_type = typename T::value_type;
        using real_res_type = std::vector<real_value_type, aligned_allocator<real_value_type, A>>;
        static constexpr std::size_t size = N;

        void load_vec(vector_type& v, const res_type& src, std::size_t i = 0) const;
        void store_vec(vector_type& v, res_type& dst, std::size_t i = 0) const;

        void load_vec_unaligned(vector_type& v, const res_type& src, std::size_t i = 0) const;
        void store_vec_unaligned(vector_type& v, res_type& dst, std::size_t i = 0) const;
    };

    template <class T, std::size_t N, std::size_t A>
    void simd_complex_tester<T, N, A>::load_vec(vector_type& v, const res_type& src, std::size_t i) const
    {
        real_res_type re(N), im(N);
        for (std::size_t j = 0; j < N; ++j)
        {
            re[j] = src[i + j].real();
            im[j] = src[i + j].imag();
        }
        v.load_aligned(&re[0], &im[0]);
    }

    template <class T, std::size_t N, std::size_t A>
    void simd_complex_tester<T, N, A>::store_vec(vector_type& v, res_type& dst, std::size_t i) const
    {
        real_res_type re(N), im(N);
        v.store_aligned(&re[0], &im[0]);
        for (std::size_t j = 0; j < N; ++j)
        {
            dst[i + j] = value_type(re[j], im[j]);
        }
    }

    template <class T, std::size_t N, std::size_t A>
    void simd_complex_tester<T, N, A>::load_vec_unaligned(vector_type& v, const res_type& src, std::size_t i) const
    {
        real_res_type re(N), im(N);
        for (std::size_t j = 0; j < N; ++j)
        {
            re[j] = src[i + j].real();
            im[j] = src[i + j].imag();
        }
        v.load_unaligned(&re[0], &im[0]);
    }

    template <class T, std::size_t N, std::size_t A>
    void simd_complex_tester<T, N, A>::store_vec_unaligned(vector_type& v, res_type& dst, std::size_t i) const
    {
        real_res_type re(N), im(N);
        v.store_unaligned(&re[0], &im[0]);
        for (std::size_t j = 0; j < N; ++j)
        {
            dst[i + j] = value_type(re[j], im[j]);
        }
    }

}

#endif
