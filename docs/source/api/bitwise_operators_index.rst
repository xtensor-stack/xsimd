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

Bitwise operators
=================

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`bitwise_not`               | per slot bitwise not                               |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`bitwise_or`                | per slot bitwise or                                |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`bitwise_xor`               | per slot bitwise xor                               |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`bitwise_and`               | per slot bitwise and                               |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`bitwise_andnot`            | per slot bitwise and not                           |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`bitwise_lshift`            | per slot bitwise and                               |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`bitwise_rshift`            | per slot bitwise and not                           |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`rotr`                      | per slot rotate right                              |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`rotl`                      | per slot rotate left                               |
+---------------------------------------+----------------------------------------------------+

----

.. doxygengroup:: batch_bitwise
   :project: xsimd
   :content-only:

