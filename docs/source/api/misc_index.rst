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


Miscellaneous
=============

Sign manipulation:

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`sign`                      | per slot sign extraction                           |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`signnz`                    | per slot sign extraction on non null elements      |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`bitofsign`                 | per slot sign bit extraction                       |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`copysign`                  | per slot sign copy                                 |
+---------------------------------------+----------------------------------------------------+

Stream operation:

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`operator<<`                | batch pretty-printing                              |
+---------------------------------------+----------------------------------------------------+

----

.. doxygengroup:: batch_miscellaneous
   :project: xsimd
   :content-only:
