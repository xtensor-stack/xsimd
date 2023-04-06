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

Comparison operators
====================

Ordering:

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`eq`                        | per slot equals to comparison                      |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`neq`                       | per slot different from comparison                 |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`gt`                        | per slot strictly greater than comparison          |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`lt`                        | per slot strictly lower than comparison            |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`ge`                        | per slot greater or equal to comparison            |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`le`                        | per slot lower or equal to comparison              |
+---------------------------------------+----------------------------------------------------+

Parity check:

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`is_even`                   | per slot check for evenness                        |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`is_odd`                    | per slot check for oddness                         |
+---------------------------------------+----------------------------------------------------+

Floating point number check:

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`isinf`                     | per slot check for infinity                        |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`isnan`                     | per slot check for NaN                             |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`isfinite`                  | per slot check for finite number                   |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`is_flint`                  | per slot check for float representing an integer   |
+---------------------------------------+----------------------------------------------------+

----

.. doxygengroup:: batch_logical
   :project: xsimd
   :content-only:
