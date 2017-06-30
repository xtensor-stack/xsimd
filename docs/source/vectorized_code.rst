.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Writing vectorized code
=======================

Assume that we have a simple function that computes the mean of two vectors, something like:

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

How can we used `xsimd` to take advantage of vectorization ?

Explicit use of an instruction set
----------------------------------

`xsimd` provides the template class ``batch<T, N>`` where ``N`` is the number of scalar values of type ``T``  involved in SIMD
instructions. If you know which intruction set is available on your machine, you can directly use the corresponding specialization
of ``batch``. For instance, assuming the AVX instruction set is available, the previous code can be vectorized the following way:

.. code::

    #include <cstddef>
    #include <vector>
    #include "xsimd/xsimd.hpp"

    void mean(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& res)
    {
        using b_type = xsimd::batch<double, 4>;
        std::size_t inc = b_type::size;
        std::size_t size = res.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        for(std::size_t i = 0; i < vec_size; i +=inc)
        {
            b_type avec(&a[i]);
            b_type bvec(&b[i]);
            b_type rvec = (avec + bvec) / 2;
            rvec.store_unaligned(&res[i]);
        }
        // Remaining part that cannot be vectorize
        for(std::size_t i = vec_size; i < size; ++i)
        {
            res[i] = (a[i] + b[i]) / 2;
        }
    }

However, if you want to write code that is portable, you cannot rely on the use of ``batch<double, 4>``.
Indeed this won't compile on a CPU where only SSE2 instruction set is available for instance. To solve this, `xsimd` provides an
auto-detection mechanism so you can use the most performant SIMD instruction set available on your hardware.

Auto detecting the instruction set
----------------------------------

Using the auto detection mechanism does not require a lot of change:

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

Aligned vs unaligned memory
---------------------------

In the previous example, you may have noticed the ``load_unaligned/store_unaligned`` functions. These
are meant for loading values from contiguous dynamically allocated memory into SIMD registers and
reciprocally. When dealing with memory transfer operations, some instructions sets required the memory
to be aligned by a given amount, others can handle both aligned and unaligned modes. In that latter case,
operating on aligned memory is always faster than operating on unaligned memory.

`xsimd` provides an aligned memory allocator which follows the standard requirements, so it can be used
with STL containers. Let's change the previous code so it can take advantage of this allocator:

.. code::

    #include <cstddef>
    #include <vector>
    #include "xsimd/xsimd.hpp"

    using vector_type = std::vector<double, XSIMD_DEFAULT_ALLOCATOR(double)>;
    void mean(const vector_type& a, const vector_type& b, vector_type& res)
    {
        using b_type = xsimd::simd_type<double>;
        std::size_t inc = b_type::size;
        std::size_t size = res.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        for(std::size_t i = 0; i < vec_size; i += inc)
        {
            b_type avec = xsimd::load_aligned(&a[i]);
            b_type bvec = xsimd::load_aligned(&b[i]);
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

Memory alignment and tag dispatching
------------------------------------

You may need to write code that can operate on any type of vectors or arrays, not only the STL ones. In that
case, you cannot make assumption on the memory alignment of the container. `xsimd` provides a tag dispatching
mechanism that allows you to easily write such a generic code:


.. code::

    #include <cstddef>
    #include <vector>
    #include "xsimd/xsimd.hpp"

    template <class C, class Tag>
    void mean(const C& a, const C& b, C& res)
    {
        using b_type = xsimd::simd_type<double>;
        std::size_t inc = b_type::size;
        std::size_t size = res.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        for(std::size_t i = 0; i < vec_size; i += inc)
        {
            b_type avec = xsimd::load_simd(&a[i], Tag());
            b_type bvec = xsimd::load_simd(&b[i], Tag());
            b_type rvec = (avec + bvec) / 2;
            xsimd::store_simd(&res[i], rvec, Tag());
        }
        // Remaining part that cannot be vectorize
        for(std::size_t i = vec_size; i < size; ++i)
        {
            res[i] = (a[i] + b[i]) / 2;
        }
    }

Here, the ``Tag`` template parameter can be ``xsimd::aligned_mode`` or ``xsimd::unaligned_mode``. Assuming the existence
of a ``get_alignment_tag`` metafunction in the code, the previous code can be invoked this way:

.. code::

    mean<get_alignment_tag<decltype(a)>>(a, b, res);

