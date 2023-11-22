.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay 

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Data transfer
=============

From memory:

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`load`                      | load values from memory                            |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`load_aligned`              | load values from aligned memory                    |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`load_unaligned`            | load values from unaligned memory                  |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`load_as`                   | load values, forcing a type conversion             |
+---------------------------------------+----------------------------------------------------+

From a scalar:

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`broadcast`                 | broadcasting a value to all slots                  |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`broadcast_as`              | broadcasting a value, forcing a type conversion    |
+---------------------------------------+----------------------------------------------------+

To memory:

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`store`                     | store values to memory                             |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`store_aligned`             | store values to aligned memory                     |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`store_unaligned`           | store values to unaligned memory                   |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`store_as`                  | store values, forcing a type conversion            |
+---------------------------------------+----------------------------------------------------+

In place:

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`swizzle`                   | rearrange slots within the batch                   |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`slide_left`                | bitwise shift the whole batch to the left          |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`slide_right`               | bitwise shift the whole batch to the right         |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`rotate_left`               | bitwise rotate the whole batch to the left         |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`rotate_right`              | bitwise rotate the whole batch to the right        |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`insert`                    | modify a single batch slot                         |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`compress`                  | pack elements according to a mask                  |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`expand`                    | select contiguous elements from the batch          |
+---------------------------------------+----------------------------------------------------+

Between batches:

+---------------------------------------+----------------------------------------------------+
| :cpp:func:`zip_lo`                    | interleave low halves of two batches               |
+---------------------------------------+----------------------------------------------------+
| :cpp:func:`zip_hi`                    | interleave high halves of two batches              |
+---------------------------------------+----------------------------------------------------+

----

.. doxygengroup:: batch_data_transfer
   :project: xsimd
   :content-only:

The following empty types are used for tag dispatching:

.. doxygenstruct:: xsimd::aligned_mode
   :project: xsimd

.. doxygenstruct:: xsimd::unaligned_mode
   :project: xsimd
