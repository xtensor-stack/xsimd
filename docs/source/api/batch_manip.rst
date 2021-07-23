.. Copyright (c) 2021, Serge Guelton

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Batch manipulation functions
============================

.. toctree::

+---------------------------------------------+---------------------------------------------------+
| :ref:`select <select-function-reference>`   | conditonal selection                              |
+---------------------------------------------+---------------------------------------------------+


.. _select-function-reference:
.. doxygenfunction:: select(batch_bool<T, A> const &cond, batch<T, A> const &true_br, batch<T, A> const &false_br)
   :project: xsimd

