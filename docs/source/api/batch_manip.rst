.. Copyright (c) 2021, Serge Guelton

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Conditional expression
======================

.. toctree::

+---------------------------------------------+---------------------------------------------------+
| :ref:`select <select-function-reference>`   | conditional selection                              |
+---------------------------------------------+---------------------------------------------------+


.. _select-function-reference:
.. doxygenfunction:: xsimd::select(batch_bool<T, A> const&, batch<T, A> const&, batch<T, A> const&)
   :project: xsimd

.. _select-function-reference:
.. doxygenfunction:: xsimd::select(batch_bool_constant<T, A> const&, batch<T, A> const&, batch<T, A> const&)
   :project: xsimd

