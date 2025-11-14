.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Batch of constants
==================

.. _xsimd-batch-constant-ref:

.. doxygenstruct:: xsimd::batch_constant
   :project: xsimd
   :members:

.. doxygenstruct:: xsimd::batch_bool_constant
   :project: xsimd
   :members:


.. doxygenfunction:: xsimd::make_batch_constant
   :project: xsimd


.. doxygenfunction:: xsimd::make_batch_bool_constant
   :project: xsimd

.. note::

   :cpp:func:`make_batch_constant` and :cpp:func:`make_batch_bool_constant` also
   accept a scalar value instead of a generator. In that case, that value is
   broadcast to each slot of the constant batch.
