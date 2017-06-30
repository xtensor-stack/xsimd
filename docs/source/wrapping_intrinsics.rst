.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Wrapping intrinsics
===================

Writing vectorized code
-----------------------

Using SIMD intrinsics provided by the compilers is the most straightforward way to write vectorized code.
For instance, assume we want to compute the mean of two vectors:

.. code::

    #include <cstddef>
    #include <vector>

    void mean(const std:vector<double>& a, const std::vector<double>& b, std::vector<double>& res)
    {
        std::size_t size = res.size();
        for(std::size_t i = 0; i < size; ++i)
        {
            res[i] = (a[i] + b[i]) / 2;
        }
    }

If we assume that the SSE2 instruction set is available, we can rewrite this code to take advantage of vectorization:

.. code::

    #include <cstddef>
    #include <vector>
    #include <emmintrin.h>

    void mean(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& res)
    {
        std::size_t size = res.size();
        // We assume that the size of the vectors is a multiple of 4.
        for(std::size_t i = 0; i < size; i += 2)
        {
            __m128d avec = _mm_loadu_pd(&a[i]);
            __m128d bvec = _mm_loadu_pd(&b[i]);
            __m128d rvec = _mm_mul_pd(_mm_add_pd(avec, bvec), _mm_set1_pd(0.5));
            _mm_storeu_pd(&res[i], rvec);
        }
    }

This code has two major drawbacks. First it is different from usual mathematical C++ code and is difficult to read. And second,
if you want to take advantage of a more recent instruction set, let's say AVX, you need to rewrite almost all the code:

.. code::

   #include <cstddef>
   #include <vector>
   #include <immintrin.h>

   void mean(const std:vector<double>& a, const std::vector<double>& b, std::vector<double>& res)
   {
       std::size_t size = res.size();
       // We assume that the size of the vectors is a multiple of 4.
       for(std::size_t i = 0; i < size; i += 4)
       {
           __m256d avec = _mm256_loadu_pd(&a[i]);
           __m256d bvec = _mm256_loadu_pd(&b[i]);
           __m256d rvec = _mm256_mul_pd(_mm256_add_pd(avec, bvec), _mm256_set1_pd(0.));
           _mm256_storeu_pd(&res[i], rvec);
        }
    }

Using wrappers
--------------

The same code using the intrinsics wrappers provided by `xsimd` looks like:

.. code::

    #include <cstddef>
    #include <vector>
    #include "xsimd/xsimd.hpp"

    void mean(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& res)
    {
        using b_type = xsimd::batch<double, 4>;
        std::size_t inc = b_type::size;
        std::size_t size = res.size();
        // We assume that the size of the vectors is a multiple of 4.
        for(std::size_t i = 0; i < size; i +=inc)
        {
            b_type avec(&a[i]);
            b_type bvec(&b[i]);
            b_type rvec = (avec + bvec) / 2;
            rvec.store_unaligned(&res[i]);
        }
    }

Now if we need to compile this code on a CPU that does support SSE2 but not AVX, the only thing to change is 

.. code::

   using b_type = xsimd::batch<double, 2>;

This is still a problem if we want to deploy our code on different architectures since we have to detect
the available instruction set and define the appropriate using type (which means a lot of ``#ifdef`` if
we want to write code that is portable). Fortunately, `xsimd` provides a way to detect the available
instruction set and thus to write generic code.

Auto-detecting the instruction set
----------------------------------

A few changes are required to benefit from the auto-detection mechanism:

.. code::

    #include <cstddef>
    #include <vector>
    #include "xsimd/xsimd.hpp"

    void mean(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& res)
    {
        using b_type = xsimd::simd_type<double>;
        std::size_t inc = b_type::size;
        std::size_t size = res.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        for(std::size_t i = 0; i < vec_size; i += inc)
        {
            b_type avec = xsimd::load_unaligned(&a[i]);
            b_type bvec = xsimd::load_unaligned(&b[i]);
            b_type rvec = (avec + bvec) / 2;
            xsimd::store_unaligned(&res[i], rvec);
            // or rvec.store_unalined(&res[i]);
        }
        // Remaining part that cannot be vectorize
        for(std::size_t i = vec_size; i < size; ++i)
        {
            res[i] = (a[i] + b[i]) / 2;
        }
    }

