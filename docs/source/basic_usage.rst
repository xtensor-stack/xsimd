.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Basic usage
===========

Manipulating abstract batches
-----------------------------

Here is an example that computes the mean of two batches, using the best
architecture available, based on compile time informations:

.. code::

    #include "xsimd/xsimd.hpp"

    namespace xs = xsimd;
    xs::batch<float> mean(xs::batch<float> lhs, xs::batch<float> rhs) {
      return (lhs + rhs) / 2;
    }

The batch can be a batch of 4 single precision floating point numbers (e.g. on
Neon) ot a batch of 8 (e.g. on AVX2).

Manipulating parametric batches
-------------------------------

The previous example can be made fully parametric, both in the batch type and
the underlying architecture. This is achieved as described in the following
example:

.. code::

    #include "xsimd/xsimd.hpp"

    namespace xs = xsimd;
    template<class T, class Arch>
    xs::batch<T, Arch> mean(xs::batch<T, Arch> lhs, xs::batch<T, Arch> rhs) {
      return (lhs + rhs) / 2;
    }

At its core, a :cpp:class:`xsimd::batch` is bound to the scalar type it contains, and to the
instruction set it can use to operate on its values.

Explicit use of an instruction set extension
--------------------------------------------

Here is an example that loads two batches of 4 double floating point values, and
computes their mean, explicitly using the AVX extension:

.. code::

    #include <iostream>
    #include "xsimd/xsimd.hpp"

    namespace xs = xsimd;

    int main(int argc, char* argv[])
    {
        xs::batch<double, xs::avx> a = {1.5, 2.5, 3.5, 4.5};
        xs::batch<double, xs::avx> b = {2.5, 3.5, 4.5, 5.5};
        auto mean = (a + b) / 2;
        std::cout << mean << std::endl;
        return 0;
    }

Note that in that case, the instruction set is explicilty specified in the batch type.

This example outputs:

.. code::

    (2.0, 3.0, 4.0, 5.0)

.. warning::

   If you allow your compiler to generate AVX2 instructions (e.g. through
   ``-mavx2``) there is nothing preventing it to optimize the above code to
   optimize the above code using AVX2 instructions.
