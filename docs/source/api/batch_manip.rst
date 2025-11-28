.. Copyright (c) 2021, Serge Guelton

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Conditional Expression
======================

+------------------------------+-------------------------------------------+
| :cpp:func:`select`           | conditional selection with mask           |
+------------------------------+-------------------------------------------+

----

.. doxygengroup:: batch_cond
   :project: xsimd
   :content-only:


In the specific case when one needs to conditionnaly increment or decrement a
batch based on a mask, :cpp:func:`incr_if` and
:cpp:func:`decr_if` provide specialized version.
