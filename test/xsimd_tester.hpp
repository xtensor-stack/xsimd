/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
* Martin Renou                                                             *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_TESTER_HPP
#define XSIMD_TESTER_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "xsimd/memory/xsimd_aligned_allocator.hpp"

namespace xsimd
{

    template <class T, std::size_t N>
    class batch;

    template <class T, std::size_t N, std::size_t A>
    struct simd_tester
    {
        using vector_type = batch<T, N>;
        using value_type = T;
        using res_type = std::vector<value_type, aligned_allocator<value_type, A>>;
        static constexpr std::size_t size = N;
    };

    namespace detail
    {
        template <class V, class S>
        void load_vec(V& vec, const S& src)
        {
            vec.load_aligned(&src[0]);
        }

        template <class V, class R>
        void store_vec(V& vec, R& res)
        {
            vec.store_aligned(&res[0]);
        }

        template <class V, class S>
        void load_vec(V& vec, const S& src, size_t i)
        {
            vec.load_aligned(&src[i]);
        }

        template <class V, class R>
        void store_vec(V& vec, R& res, size_t i)
        {
            vec.store_aligned(&res[i]);
        }
    }
}

#endif
