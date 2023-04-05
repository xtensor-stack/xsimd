.. Copyright (c) 2021, Serge Guelton

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Conditional expression
======================

+------------------------------+-------------------------------------------+
| :cpp:func:`select`           | conditional selection with mask           |
+------------------------------+-------------------------------------------+

----

.. doxygenfunction:: select(batch_bool<T, A> const &cond, batch<T, A> const &true_br, batch<T, A> const &false_br) noexcept
   :project: xsimd

.. doxygenfunction:: select(batch_bool_constant<batch<T, A>, Values...> const &cond, batch<T, A> const &true_br, batch<T, A> const &false_br) noexcept
   :project: xsimd


In the specific case when one needs to conditionnaly increment or decrement a
batch based on a mask, :cpp:func:`incr_if` and
:cpp:func:`decr_if` provide specialized version.
