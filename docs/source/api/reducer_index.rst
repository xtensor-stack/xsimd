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

Reduction operators
===================

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`reduce`                    | generic batch reduction                            |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`reduce_add`                | sum of each batch element                          |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`reduce_max`                | max of the batch elements                          |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`reduce_min`                | min of the batch elements                          |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`haddp`                     | horizontal sum across batches                      |
+---------------------------------------+----------------------------------------------------+

----

.. doxygengroup:: batch_reducers
   :project: xsimd
   :content-only:
