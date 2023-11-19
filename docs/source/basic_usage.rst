.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Basic usage
===========

Manipulating abstract batches
-----------------------------

Here is an example that computes the mean of two batches, using the best
architecture available, based on compile time informations:

.. literalinclude:: ../../test/doc/manipulating_abstract_batches.cpp

The batch can be a batch of 4 single precision floating point numbers (e.g. on
Neon) or a batch of 8 (e.g. on AVX2).

Manipulating parametric batches
-------------------------------

The previous example can be made fully parametric, both in the batch type and
the underlying architecture. This is achieved as described in the following
example:

.. literalinclude:: ../../test/doc/manipulating_parametric_batches.cpp

At its core, a :cpp:class:`xsimd::batch` is bound to the scalar type it contains, and to the
instruction set it can use to operate on its values.

Explicit use of an instruction set extension
--------------------------------------------

Here is an example that loads two batches of 4 double floating point values, and
computes their mean, explicitly using the AVX extension:

.. literalinclude:: ../../test/doc/explicit_use_of_an_instruction_set.cpp

Note that in that case, the instruction set is explicilty specified in the batch type.

This example outputs:

.. code::

   (2.0, 3.0, 4.0, 5.0)

.. warning::

   If you allow your compiler to generate AVX2 instructions (e.g. through
   ``-mavx2``) there is nothing preventing it from optimizing the above code
   using AVX2 instructions.
