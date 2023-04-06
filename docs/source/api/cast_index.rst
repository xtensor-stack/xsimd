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


Type conversion
===============

Cast:

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`batch_cast`                | ``static_cast`` on batch types                     |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`bitwise_cast`              | ``reinterpret_cast`` on batch types                |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`batch_bool_cast`           | ``static_cast`` on batch predicate types           |
+---------------------------------------+----------------------------------------------------+

Conversion:

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`to_float`                  | per slot conversion to floating point              |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`to_int`                    | per slot conversion to integer                     |
+---------------------------------------+----------------------------------------------------+

----

.. doxygengroup:: batch_conversion
   :project: xsimd
   :content-only:
