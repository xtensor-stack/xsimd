.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay 

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. raw:: html

   <style>
   .rst-content table.docutils {
       width: 100%;
       table-layout: fixed;
   }

   table.docutils .line-block {
       margin-left: 0;
       margin-bottom: 0;
   }

   table.docutils code.literal {
       color: initial;
   }

   code.docutils {
       background: initial;
   }
   </style>


.. _Arch Dispatching:

Arch Dispatching
================

`xsimd` provides a generic way to dispatch a function call based on the architecture the code was compiled for and the architectures available at runtime.
The :cpp:func:`xsimd::dispatch` function takes a functor whose call operator takes an architecture parameter as first operand, followed by any number of arguments ``Args...`` and turn it into a
dispatching functor that takes ``Args...`` as arguments.

.. doxygenfunction:: xsimd::dispatch
    :project: xsimd

Following code showcases a usage of the :cpp:func:`xsimd::dispatch` function:

.. code-block:: c++

    #include "sum.hpp"

    // Create the dispatching function, specifying the architecture we want to
    // target.
    auto dispatched = xsimd::dispatch<xsimd::arch_list<xsimd::avx2, xsimd::sse2>>(sum{});

    // Call the appropriate implementation based on runtime information.
    float res = dispatched(data, 17);

This code does *not* require any architecture-specific flags. The architecture
specific details follow.

The ``sum.hpp`` header contains the function being actually called, in an
architecture-agnostic description:

.. literalinclude:: ../../../test/doc/sum.hpp


The SSE2 and AVX2 version needs to be provided in other compilation units, compiled with the appropriate flags, for instance:

.. literalinclude:: ../../../test/doc/sum_avx2.cpp

.. literalinclude:: ../../../test/doc/sum_sse2.cpp

